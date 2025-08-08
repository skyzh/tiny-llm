#include <cstdint>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext_ref {
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

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: output dtype must be float32");
    }

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

    // Launch the CPU kernel
    encoder.dispatch([out_ptr = out.data<float>(), out_shape = out.shape(), q = mx::array::unsafe_weak_copy(q),
                      k = mx::array::unsafe_weak_copy(k), v = mx::array::unsafe_weak_copy(v),
                      mask = mx::array::unsafe_weak_copy(mask), num_heads = num_heads_, num_kv_heads = num_kv_heads_,
                      scale = scale_]() {
        const int64_t N = q.shape()[0];
        const int64_t L = q.shape()[1];
        const int64_t S = k.shape()[1];
        const int64_t E = q.shape()[2];
        const int64_t N_Q_HEAD = L * E;
        const int64_t N_K_HEAD = S * E;
        const int64_t Br = 32;
        const int64_t Bc = 32;
        const int64_t Tr = (L + Br - 1) / Br;
        const int64_t Tc = (S + Bc - 1) / Bc;

        const int64_t q_kv_heads_ratio = num_heads / num_kv_heads;
        const float *q_ptr = q.data<float>();
        const float *k_ptr = k.data<float>();
        const float *v_ptr = v.data<float>();
        const float *m_ptr = mask.data<float>();

        for (int64_t n = 0; n < N; n++) {
            const float *q_batch = q_ptr + n * N_Q_HEAD;
            const float *k_batch = k_ptr + (n / q_kv_heads_ratio) * N_K_HEAD;
            const float *v_batch = v_ptr + (n / q_kv_heads_ratio) * N_K_HEAD;
            for (int64_t i = 0; i < Tr; i++) {
                std::vector<float> q_i(Br * E, 0.0);
                int br_upper_bound = std::min(L - i * Br, Br);
                // Load Qi
                for (int64_t a = 0; a < br_upper_bound; a++) {
                    for (int64_t b = 0; b < E; b++) {
                        int q_idx = (i * Br + a) * E + b;
                        q_i[a * E + b] = q_batch[q_idx];
                    }
                }
                std::vector<float> o_i(Br * E, 0.0);
                std::vector<float> l_i(Br, 0.0);
                std::vector<float> m_i(Br, -std::numeric_limits<float>::infinity());
                for (int64_t j = 0; j < Tc; j++) {
                    int bc_upper_bound = std::min(S - j * Bc, Bc);
                    // Each kernel processes a block of Br x Bc
                    // Load Kj and Vj
                    std::vector<float> k_j(Bc * E, 0.0);
                    std::vector<float> v_j(Bc * E, 0.0);
                    for (int64_t a = 0; a < bc_upper_bound; a++) {
                        int64_t kv_idx_base = j * Bc + a;
                        for (int64_t b = 0; b < E; b++) {
                            int kv_idx = kv_idx_base * E + b;
                            if (kv_idx_base < S) {
                                k_j[a * E + b] = k_batch[kv_idx];
                                v_j[a * E + b] = v_batch[kv_idx];
                            }
                        }
                    }

                    std::vector<float> s_i(Br * Bc, 0.0);
                    // Compute s_i = q_i * k_j^T
                    for (int64_t a = 0; a < br_upper_bound; a++) {
                        for (int64_t b = 0; b < bc_upper_bound; b++) {
                            for (int64_t c = 0; c < E; c++) {
                                s_i[a * Bc + b] += q_i[a * E + c] * k_j[b * E + c];
                            }
                        }
                    }

                    // Add mask and scale
                    for (int64_t a = 0; a < br_upper_bound; a++) {
                        for (int64_t b = 0; b < bc_upper_bound; b++) {
                            int m_idx_1 = n;
                            int m_idx_2 = i * Br + a;
                            int m_idx_3 = j * Bc + b;
                            int m_idx_converted = mx::elem_to_loc(m_idx_1 * L * S + m_idx_2 * S + m_idx_3, mask);
                            s_i[a * Bc + b] *= scale;
                            s_i[a * Bc + b] += m_ptr[m_idx_converted];
                        }
                    }

                    // m_i from iteration j = max(m_i from iteration j-1, rowmax(s_i))
                    std::vector<float> m_i_diff(Br, 0.0);
                    for (int64_t a = 0; a < br_upper_bound; a++) {
                        float rowmax = -std::numeric_limits<float>::infinity();
                        for (int64_t b = 0; b < bc_upper_bound; b++) {
                            rowmax = std::max(rowmax, s_i[a * Bc + b]);
                        }
                        float max = std::max(m_i[a], rowmax);
                        m_i_diff[a] = m_i[a] - max;
                        m_i[a] = max;
                    }

                    // compute p_j
                    std::vector<float> p(Br * Bc, 0.0);
                    for (int64_t a = 0; a < br_upper_bound; a++) {
                        for (int64_t b = 0; b < bc_upper_bound; b++) {
                            p[a * Bc + b] = std::exp(s_i[a * Bc + b] - m_i[a]);
                        }
                    }

                    // compute l
                    for (int64_t a = 0; a < br_upper_bound; a++) {
                        // compute rowsum(p)
                        float rowsum = 0.0;
                        for (int64_t b = 0; b < bc_upper_bound; b++) {
                            rowsum += p[a * Bc + b];
                        }
                        l_i[a] = std::exp(m_i_diff[a]) * l_i[a] + rowsum;
                    }

                    // compute o_i = diag(std::exp(m_i_diff)) * o_i from prev iteration + p * v_j
                    for (int64_t a = 0; a < br_upper_bound; a++) {
                        for (int64_t c = 0; c < E; c++) {
                            // compute p @ v_j
                            float res = 0;
                            for (int64_t b = 0; b < bc_upper_bound; b++) {
                                res += p[a * Bc + b] * v_j[b * E + c];
                            }
                            o_i[a * E + c] = std::exp(m_i_diff[a]) * o_i[a * E + c] + res;
                        }
                    }
                }
                // o_i = diag(l_i)^-1 * o_i
                for (int64_t a = 0; a < br_upper_bound; a++) {
                    for (int64_t b = 0; b < E; b++) {
                        o_i[a * E + b] /= l_i[a];
                    }
                }
                // l_i = m_i + log(l_i)
                for (int64_t a = 0; a < br_upper_bound; a++) {
                    l_i[a] = m_i[a] + std::log(l_i[a]);
                }
                // store o_i
                for (int64_t a = 0; a < br_upper_bound; a++) {
                    for (int64_t b = 0; b < E; b++) {
                        int out_idx = i * Br + a;
                        if (out_idx < L) {
                            out_ptr[n * N_Q_HEAD + out_idx * E + b] = o_i[a * E + b];
                        }
                    }
                }
                // ignore l_i -- we might use it in the future
            }
        }
    });
}

void FlashAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &mask = inputs[3];
    auto &out = outputs[0];

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: output dtype must be float32");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &s = stream();
    auto &d = mx::metal::device(s.device);

    // Make a kernel from this metal library
    auto library = d.get_library("tiny_llm_ext_ref");
    auto kernel = d.get_kernel("flash_attention_f32_e128", library);

    // Prepare to encode kernel
    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    // Encode input arrays to kernel
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_input_array(mask, 3);
    compute_encoder.set_output_array(out, 4);    
    compute_encoder.set_vector_bytes(mask.shape(), 5);
    compute_encoder.set_vector_bytes(mask.strides(), 6);

    if (!q.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: q must be contiguous");
    }
    if (!k.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: k must be contiguous");
    }
    if (!v.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: v must be contiguous");
    }
    if (!out.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: out must be contiguous");
    }

    const int N = q.shape()[0];
    const int L = q.shape()[1];
    const int S = k.shape()[1];
    const int E = q.shape()[2];

    compute_encoder.set_bytes(N, 7);
    compute_encoder.set_bytes(L, 8);
    compute_encoder.set_bytes(S, 9);
    compute_encoder.set_bytes(E, 10);

    // Make sure the data type matches with the metal kernel: otherwise you'll get flaky issues and stuck :(
    compute_encoder.set_bytes(num_kv_heads_, 11);
    compute_encoder.set_bytes(num_heads_, 12);
    compute_encoder.set_bytes(scale_, 13);

    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    size_t simd_width = kernel->threadExecutionWidth();

    const int Br = 32;
    const int Bc = 32;
    if (simd_width * Br > tgp_size) {
        throw std::runtime_error("flash_attention: simd_width * Br must be equal to tgp_size");
    }
    if (Bc > simd_width) {
        throw std::runtime_error("flash_attention: Bc must be less than simd_width");
    }

    if (E > 128) {
        throw std::runtime_error("flash_attention: E must be less than 128");
    }

    if (Br > 32) {
        throw std::runtime_error("flash_attention: Br must be less than 32");
    }

    if (Bc > 32) {
        throw std::runtime_error("flash_attention: Bc must be less than 32");
    }

    const int Tr = (L + Br - 1) / Br;
    const int Tc = (S + Bc - 1) / Bc;

    compute_encoder.set_bytes(Br, 14);
    compute_encoder.set_bytes(Bc, 15);
    compute_encoder.set_bytes(Tr, 16);
    compute_encoder.set_bytes(Tc, 17);

    MTL::Size num_threadgroups = MTL::Size(N, Tr, 1);
    MTL::Size num_threads_per_group = MTL::Size(Br, simd_width, 1);

    compute_encoder.dispatch_threadgroups(num_threadgroups, num_threads_per_group);
}
}  // namespace tiny_llm_ext_ref
