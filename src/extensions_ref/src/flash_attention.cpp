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
                          const float scale, const int mask_mode, const int num_kv_heads, const int num_heads,
                          mx::StreamOrDevice s) {
    if ((q.dtype() != mx::float32 && q.dtype() != mx::bfloat16) || k.dtype() != q.dtype() || v.dtype() != q.dtype() ||
        mask.dtype() != mx::float32) {
        throw std::runtime_error(
            "flash_attention: q, k, and v must have the same float32 or bfloat16 dtype; mask must be float32");
    }
    if (q.shape().size() != 3 || k.shape().size() != 3 || v.shape().size() != 3) {
        throw std::runtime_error("flash_attention: all input arrays must be 3D");
    }
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: num_heads must be divisible by num_kv_heads");
    }
    if (mask_mode < 0 || mask_mode > 2) {
        throw std::runtime_error("flash_attention: mask_mode must be 0 (none), 1 (causal), or 2 (additive)");
    }
    if (mask_mode == 2 && mask.shape().size() != 3) {
        throw std::runtime_error("flash_attention: an additive mask must be 3D");
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
    if (k.shape()[0] != v.shape()[0]) {
        throw std::runtime_error("flash_attention: k and v batch/head dimensions must match");
    }
    if (mask_mode == 2 &&
        (mask.shape()[0] != q.shape()[0] || mask.shape()[1] != q.shape()[1] || mask.shape()[2] != k.shape()[1])) {
        throw std::runtime_error("flash_attention: mask must be broadcastable to q, k, v");
    }

    return mx::array(q.shape(), q.dtype(),
                     std::make_shared<FlashAttention>(to_stream(s), scale, mask_mode, num_kv_heads, num_heads),
                     {q, k, v, mask});
}

void FlashAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &q = inputs[0];
    auto &k = inputs[1];
    auto &v = inputs[2];
    auto &mask = inputs[3];
    auto &out = outputs[0];

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: the CPU path only supports float32");
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
                      scale = scale_, mask_mode = mask_mode_]() {
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
                const int64_t causal_offset = S - L;
                for (int64_t j = 0; j < Tc; j++) {
                    int64_t row_max = i * Br + br_upper_bound - 1;
                    int64_t col_min = j * Bc;
                    // Causal masking: if the entire block of K is masked out by causal mask, we can skip the
                    // computation for this block.
                    if (mask_mode == 1 && col_min > row_max + causal_offset) {
                        continue;
                    }
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
                            s_i[a * Bc + b] *= scale;
                            if (mask_mode == 1 && j * Bc + b > i * Br + a + causal_offset) {
                                s_i[a * Bc + b] = -std::numeric_limits<float>::infinity();
                            } else if (mask_mode == 2) {
                                int m_idx_1 = n;
                                int m_idx_2 = i * Br + a;
                                int m_idx_3 = j * Bc + b;
                                int m_idx_converted = mx::elem_to_loc(m_idx_1 * L * S + m_idx_2 * S + m_idx_3, mask);
                                s_i[a * Bc + b] += m_ptr[m_idx_converted];
                            }
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
                        m_i_diff[a] = m_i[a] == -std::numeric_limits<float>::infinity()
                                          ? -std::numeric_limits<float>::infinity()
                                          : m_i[a] - max;
                        m_i[a] = max;
                    }

                    // compute p_j
                    std::vector<float> p(Br * Bc, 0.0);
                    for (int64_t a = 0; a < br_upper_bound; a++) {
                        for (int64_t b = 0; b < bc_upper_bound; b++) {
                            p[a * Bc + b] = s_i[a * Bc + b] == -std::numeric_limits<float>::infinity()
                                                ? 0.0f
                                                : std::exp(s_i[a * Bc + b] - m_i[a]);
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

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &s = stream();
    auto &d = mx::metal::device(s.device);

    auto library = d.get_library("tiny_llm_ext_ref");
    auto &compute_encoder = d.get_command_encoder(s.index);

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

    if (E <= 0 || E > 128) {
        throw std::runtime_error("flash_attention: E must be in the range [1, 128]");
    }

    if (E == 128 && q.dtype() == mx::bfloat16) {
        auto kernel = d.get_kernel("flash_attention_mma_bf16_d128", library);
        compute_encoder.set_compute_pipeline_state(kernel);
        compute_encoder.set_input_array(q, 0);
        compute_encoder.set_input_array(k, 1);
        compute_encoder.set_input_array(v, 2);
        compute_encoder.set_input_array(mask, 3);
        compute_encoder.set_output_array(out, 4);
        compute_encoder.set_bytes(mask_mode_, 5);
        compute_encoder.set_bytes(N, 6);
        compute_encoder.set_bytes(L, 7);
        compute_encoder.set_bytes(S, 8);
        compute_encoder.set_bytes(num_kv_heads_, 9);
        compute_encoder.set_bytes(num_heads_, 10);
        compute_encoder.set_bytes(scale_, 11);

        const int batch_size = N / num_heads_;
        const int query_blocks = (L + 63) / 64;
        compute_encoder.dispatch_threadgroups(MTL::Size(query_blocks, num_heads_, batch_size), MTL::Size(32, 8, 1));
        return;
    }

    if (q.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: bfloat16 GPU inputs require E == 128");
    }

    auto kernel = d.get_kernel("flash_attention_scalar_f32", library);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_input_array(mask, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_bytes(mask_mode_, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(L, 7);
    compute_encoder.set_bytes(S, 8);
    compute_encoder.set_bytes(E, 9);
    compute_encoder.set_bytes(num_kv_heads_, 10);
    compute_encoder.set_bytes(num_heads_, 11);
    compute_encoder.set_bytes(scale_, 12);

    const int query_blocks = (L + 15) / 16;
    compute_encoder.dispatch_threadgroups(MTL::Size(N, query_blocks, 1), MTL::Size(32, 16, 1));
}
}  // namespace tiny_llm_ext_ref
