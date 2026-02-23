#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {
mx::array flash_attention(const mx::array &q, const mx::array &k, const mx::array &v, const mx::array &mask,
                          const float scale, const int num_kv_heads, const int num_heads, mx::StreamOrDevice s) {
    if (q.dtype() != mx::float32 || k.dtype() != mx::float32 || v.dtype() != mx::float32 || mask.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: all input arrays must be float32");
    }
    if (q.shape().size() != 3 || k.shape().size() != 3 || v.shape().size() != 3) {
        throw std::runtime_error("flash_attention: all input arrays must be 3D");
    }
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: num_heads must be divisible by num_kv_heads");
    }
    if (mask.shape().size() != 3) {
        throw std::runtime_error("flash_attention: mask must be 3D");
    }

    // Q: [N, L, E]
    // K: [N_KV, S, E]
    // V: [N_KV, S, E]
    // O: [N, L, E]
    // M: [N, L, S] (optional, needs broadcasting)

    if (q.shape()[0] % num_heads != 0) {
        throw std::runtime_error("flash_attention: q.shape[0] must be divisible by num_heads");
    }
    if (k.shape()[0] % num_kv_heads != 0 || v.shape()[0] % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: k.shape[0] and v.shape[0] must be divisible by num_kv_heads");
    }
    if (q.shape()[2] != k.shape()[2] || q.shape()[2] != v.shape()[2]) {
        throw std::runtime_error("flash_attention: q.shape[2] must be equal to k.shape[2] and v.shape[2]");
    }
    if (q.shape()[0] / num_heads != k.shape()[0] / num_kv_heads) {
        throw std::runtime_error("flash_attention: number of heads mismatch");
    }
    if (k.shape()[1] != v.shape()[1]) {
        throw std::runtime_error("flash_attention: k.shape[1] must be equal to v.shape[1]");
    }
    if (mask.shape()[0] != q.shape()[0] || mask.shape()[1] != q.shape()[1] || mask.shape()[2] != k.shape()[1]) {
        throw std::runtime_error("flash_attention: mask must be broadcastable to q, k, v");
    }

    return mx::array(q.shape(), mx::float32,
                     std::make_shared<FlashAttention>(to_stream(s), scale, num_kv_heads, num_heads), {q, k, v, mask});
}

void FlashAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &q = inputs[0];
    auto &k = inputs[1];
    auto &v = inputs[2];
    auto &mask = inputs[3];
    auto &out = outputs[0];

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_input_array(mask);
    encoder.set_output_array(out);

    if (!q.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: q must be contiguous");
    }
    if (!k.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: k must be contiguous");
    }
    if (!v.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: v must be contiguous");
    }

    encoder.dispatch([
        out_ptr = out.data<float>(),
        out_shape = out.shape(),
        q = mx::array::unsafe_weak_copy(q),
        k = mx::array::unsafe_weak_copy(k),
        v = mx::array::unsafe_weak_copy(v),
        mask = mx::array::unsafe_weak_copy(mask),
        scale = scale_,
        num_heads = num_heads_,
        num_kv_heads = num_kv_heads_
    ] {
        // Q: [N, L, E]
        // K: [N_KV, S, E]
        // V: [N_KV, S, E]

        const int N = q.shape()[0];
        const int L = q.shape()[1];
        const int S = k.shape()[1];
        const int E = q.shape()[2];

        const int Br = 32;
        const int Bc = 32;
        const int Tr = (L + Br - 1) / Br;
        const int Tc = (S + Bc - 1) / Bc;

        const int q_kv_heads_ratio = num_heads / num_kv_heads;
        const float *q_ptr = q.data<float>();
        const float *k_ptr = k.data<float>();
        const float *v_ptr = v.data<float>();
        const float *mask_ptr = mask.data<float>();

        for (int n = 0; n < N; n++) {
            const float *q_batch = q_ptr + n * L * E;
            const float *k_batch = k_ptr + (n / q_kv_heads_ratio) * S * E;
            const float *v_batch = v_ptr + (n / q_kv_heads_ratio) * S * E;
            const float *mask_batch = mask_ptr + n * L * S;
            float *out_batch = out_ptr + n * L * E;
            for (int i = 0; i < Tr; i++) {
                // Divide L into blocks of size Br
                std::vector<float> q_i(Br * E, 0.0);
                // Load q_i
                // Why load into a separate buffer? We need to reuse q_i for every block of K and V, 
                // and it's more efficient to load once than to read from global memory repeatedly.
                int br_upper_bound = std::min(L - i * Br, Br);
                for (int a = 0; a < br_upper_bound; a++) {
                    for (int b = 0; b < E; b++) {
                        q_i[a * E + b] = q_batch[(i * Br + a) * E + b];
                    }
                }

                std::vector<float> m_i(Br, -std::numeric_limits<float>::infinity());
                std::vector<float> p_i(Br * Bc, 0.0);
                std::vector<float> l_i(Br, 0.0);
                std::vector<float> o_i(Br * E, 0.0);

                for (int j = 0; j < Tc; j++) {
                    // Divide S into blocks of size Bc
                    std::vector<float> k_j(Bc * E, 0.0); // should consider tranpose
                    std::vector<float> v_j(Bc * E, 0.0);
                    // Load k_j and v_j
                    int bc_upper_bound = std::min(S - j * Bc, Bc);
                    for (int a = 0; a < bc_upper_bound; a++) {
                        for (int b = 0; b < E; b++) {
                            k_j[a * E + b] = k_batch[(j * Bc + a) * E + b];
                            v_j[a * E + b] = v_batch[(j * Bc + a) * E + b];
                        }
                    }

                    // Compute matmul for s_i = q_i * k_j^T : [Br, E] x [E, Bc] -> [Br, Bc]
                    std::vector<float> s_i(Br * Bc, 0.0);
                    for (int a = 0; a < br_upper_bound; a++) {
                        for (int b = 0; b < bc_upper_bound; b++) {
                            for (int c = 0; c < E; c++) {
                                s_i[a * Bc + b] += (q_i[a * E + c] * k_j[b * E + c]);
                            }
                            s_i[a * Bc + b] *= scale;
                            s_i[a * Bc + b] += mask_batch[(i * Br + a) * S + j * Bc + b];
                        }
                    }             

                    // Online softmax
                    // compute m_i = max(m_i, s_i)
                    std::vector<float> m_i_diff(Br, 0.0);
                    for (int a = 0; a < br_upper_bound; a++) {
                        float rowmax = -std::numeric_limits<float>::infinity();
                        for (int b = 0; b < bc_upper_bound; b++) {
                            rowmax = std::max(rowmax, s_i[a * Bc + b]);
                        }
                        m_i_diff[a] = m_i[a] - rowmax;
                        m_i[a] = std::max(m_i[a], rowmax);
                    }

                    // compute p_i = exp(s_i - m_i)
                    for (int a = 0; a < br_upper_bound; a++) {
                        for (int b = 0; b < bc_upper_bound; b++) {
                            p_i[a * Bc + b] = std::exp(s_i[a * Bc + b] - m_i[a]);
                        }
                    }

                    // compute l_i = exp(m_i_diff) * l_i + sum(p_i)
                    for (int a = 0; a < br_upper_bound; a++) {
                        float rowsum = 0.0;
                        for (int b = 0; b < bc_upper_bound; b++) {
                            rowsum += p_i[a * Bc + b];
                        }
                        l_i[a] = std::exp(m_i_diff[a]) * l_i[a] + rowsum;
                    }

                    // compute o_i = diag(exp(m_i_diff)) * o_i from prev iteration + p_i * v_j
                    for (int a = 0; a < br_upper_bound; a++) {
                        for (int b = 0; b < E; b++) {
                            o_i[a * E + b] = std::exp(m_i_diff[a]) * o_i[a * E + b];
                            // compute p_i * v_j
                            for (int c = 0; c < bc_upper_bound; c++) {
                                o_i[a * E + b] += p_i[a * Bc + c] * v_j[c * E + b];
                            }
                        }
                    }
                }

                // compute finial o_i
                for (int a = 0; a < br_upper_bound; a++) {
                    for (int b = 0; b < E; b++) {
                        o_i[a * E + b] /= l_i[a];
                    }
                }

                // store o_i to out
                for (int a = 0; a < br_upper_bound; a++) {
                    for (int b = 0; b < E; b++) {
                        out_batch[(i * Br + a) * E + b] = o_i[a * E + b];
                    }
                }
            }
        }
    });
}

void FlashAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &q = inputs[0];
    auto &k = inputs[1];
    auto &v = inputs[2];
    auto &mask = inputs[3];
    auto &scale = inputs[4];

    auto &out = outputs[0];

    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto lib = d.get_library("tiny_llm_ext");
    auto kernel = d.get_kernel("flash_attention", lib);
    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_input_array(mask, 3);
    compute_encoder.set_input_array(scale, 4);
    compute_encoder.set_output_array(out, 5);

}

}  // namespace tiny_llm_ext
