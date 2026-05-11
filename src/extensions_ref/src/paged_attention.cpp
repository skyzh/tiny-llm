#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext_ref {

mx::array paged_attention(const mx::array &q, const mx::array &key_pages, const mx::array &value_pages,
                          const mx::array &block_table, const mx::array &context_lens, const float scale,
                          const bool is_causal, const int num_kv_heads, const int num_heads, mx::StreamOrDevice s) {
    if (q.dtype() != mx::float32 || key_pages.dtype() != mx::float32 || value_pages.dtype() != mx::float32) {
        throw std::runtime_error("paged_attention: q, key_pages, and value_pages must be float32");
    }
    if (block_table.dtype() != mx::int32 || context_lens.dtype() != mx::int32) {
        throw std::runtime_error("paged_attention: block_table and context_lens must be int32");
    }
    if (q.shape().size() != 3) {
        throw std::runtime_error("paged_attention: q must be 3D [B * H_q, L, D]");
    }
    if (key_pages.shape().size() != 4 || value_pages.shape().size() != 4) {
        throw std::runtime_error("paged_attention: page tensors must be 4D [P, H_kv, page_size, D]");
    }
    if (block_table.shape().size() != 2 || context_lens.shape().size() != 1) {
        throw std::runtime_error("paged_attention: block_table must be 2D and context_lens must be 1D");
    }
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("paged_attention: num_heads must be divisible by num_kv_heads");
    }
    if (q.shape()[0] % num_heads != 0) {
        throw std::runtime_error("paged_attention: q.shape[0] must be divisible by num_heads");
    }
    if (key_pages.shape() != value_pages.shape()) {
        throw std::runtime_error("paged_attention: key_pages and value_pages must have the same shape");
    }
    if (key_pages.shape()[1] != num_kv_heads) {
        throw std::runtime_error("paged_attention: page tensor head count must equal num_kv_heads");
    }
    if (q.shape()[2] != key_pages.shape()[3]) {
        throw std::runtime_error("paged_attention: q and page tensors must have the same head dimension");
    }
    if (block_table.shape()[0] != context_lens.shape()[0]) {
        throw std::runtime_error("paged_attention: block_table and context_lens batch sizes must match");
    }
    if (q.shape()[0] / num_heads != block_table.shape()[0]) {
        throw std::runtime_error("paged_attention: q batch size must match block_table batch size");
    }

    return mx::array(q.shape(), mx::float32,
                     std::make_shared<PagedAttention>(to_stream(s), scale, is_causal, num_kv_heads, num_heads),
                     {q, key_pages, value_pages, block_table, context_lens});
}

void PagedAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &q = inputs[0];
    auto &key_pages = inputs[1];
    auto &value_pages = inputs[2];
    auto &block_table = inputs[3];
    auto &context_lens = inputs[4];
    auto &out = outputs[0];

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("paged_attention: output dtype must be float32");
    }
    if (!q.flags().row_contiguous || !key_pages.flags().row_contiguous || !value_pages.flags().row_contiguous ||
        !block_table.flags().row_contiguous || !context_lens.flags().row_contiguous) {
        throw std::runtime_error("paged_attention: all inputs must be contiguous");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(q);
    encoder.set_input_array(key_pages);
    encoder.set_input_array(value_pages);
    encoder.set_input_array(block_table);
    encoder.set_input_array(context_lens);
    encoder.set_output_array(out);

    encoder.dispatch([out_ptr = out.data<float>(), q = mx::array::unsafe_weak_copy(q),
                      key_pages = mx::array::unsafe_weak_copy(key_pages),
                      value_pages = mx::array::unsafe_weak_copy(value_pages),
                      block_table = mx::array::unsafe_weak_copy(block_table),
                      context_lens = mx::array::unsafe_weak_copy(context_lens), scale = scale_,
                      is_causal = is_causal_, num_kv_heads = num_kv_heads_, num_heads = num_heads_]() {
        const int64_t N = q.shape()[0];
        const int64_t L = q.shape()[1];
        const int64_t D = q.shape()[2];
        const int64_t page_size = key_pages.shape()[2];
        const int64_t max_pages = block_table.shape()[1];
        const int64_t q_kv_ratio = num_heads / num_kv_heads;

        const float *q_ptr = q.data<float>();
        const float *key_ptr = key_pages.data<float>();
        const float *value_ptr = value_pages.data<float>();
        const int32_t *block_ptr = block_table.data<int32_t>();
        const int32_t *lens_ptr = context_lens.data<int32_t>();

        const int64_t Br = 32;
        const int64_t Bc = 32;
        const int64_t Tr = (L + Br - 1) / Br;
        const int64_t S_capacity = max_pages * page_size;
        const int64_t context_capacity = std::max<int64_t>(S_capacity, 0);

        for (int64_t n = 0; n < N; n++) {
            const int64_t batch = n / num_heads;
            const int64_t q_head = n % num_heads;
            const int64_t kv_head = q_head / q_kv_ratio;
            const int64_t context_len = lens_ptr[batch];
            if (context_len <= 0) {
                std::fill(out_ptr + n * L * D, out_ptr + (n + 1) * L * D, 0.0f);
                continue;
            }

            const int64_t context_limit = std::min<int64_t>(context_len, context_capacity);
            const int64_t Tc = (context_limit + Bc - 1) / Bc;
            const int64_t causal_offset = context_len - L;

            for (int64_t i = 0; i < Tr; i++) {
                const int64_t br_upper_bound = std::min<int64_t>(L - i * Br, Br);
                std::vector<float> q_i(Br * D, 0.0f);
                for (int64_t a = 0; a < br_upper_bound; a++) {
                    for (int64_t c = 0; c < D; c++) {
                        q_i[a * D + c] = q_ptr[n * L * D + (i * Br + a) * D + c];
                    }
                }

                std::vector<float> o_i(Br * D, 0.0f);
                std::vector<float> l_i(Br, 0.0f);
                std::vector<float> m_i(Br, -std::numeric_limits<float>::infinity());

                for (int64_t j = 0; j < Tc; j++) {
                    const int64_t col_min = j * Bc;
                    const int64_t row_max = i * Br + br_upper_bound - 1;
                    if (is_causal && col_min > row_max + causal_offset) {
                        continue;
                    }

                    const int64_t bc_upper_bound = std::min<int64_t>(context_limit - col_min, Bc);
                    std::vector<float> s_i(Br * Bc, -std::numeric_limits<float>::infinity());
                    std::vector<char> valid(Br * Bc, 0);

                    for (int64_t b = 0; b < bc_upper_bound; b++) {
                        const int64_t key_pos = col_min + b;
                        const int64_t page_idx = key_pos / page_size;
                        if (page_idx >= max_pages) {
                            continue;
                        }
                        const int64_t page_id = block_ptr[batch * max_pages + page_idx];
                        if (page_id < 0) {
                            continue;
                        }
                        const int64_t slot = key_pos - page_idx * page_size;
                        for (int64_t a = 0; a < br_upper_bound; a++) {
                            if (is_causal && key_pos > i * Br + a + causal_offset) {
                                continue;
                            }
                            float score = 0.0f;
                            for (int64_t c = 0; c < D; c++) {
                                const int64_t k_idx = ((page_id * num_kv_heads + kv_head) * page_size + slot) * D + c;
                                score += q_i[a * D + c] * key_ptr[k_idx];
                            }
                            s_i[a * Bc + b] = score * scale;
                            valid[a * Bc + b] = 1;
                        }
                    }

                    for (int64_t a = 0; a < br_upper_bound; a++) {
                        float rowmax = -std::numeric_limits<float>::infinity();
                        for (int64_t b = 0; b < bc_upper_bound; b++) {
                            if (valid[a * Bc + b]) {
                                rowmax = std::max(rowmax, s_i[a * Bc + b]);
                            }
                        }
                        if (rowmax == -std::numeric_limits<float>::infinity()) {
                            continue;
                        }

                        const float new_max = std::max(m_i[a], rowmax);
                        const float old_scale = std::exp(m_i[a] - new_max);
                        float rowsum = 0.0f;
                        for (int64_t b = 0; b < bc_upper_bound; b++) {
                            if (valid[a * Bc + b]) {
                                rowsum += std::exp(s_i[a * Bc + b] - new_max);
                            }
                        }

                        for (int64_t c = 0; c < D; c++) {
                            float res = 0.0f;
                            for (int64_t b = 0; b < bc_upper_bound; b++) {
                                if (!valid[a * Bc + b]) {
                                    continue;
                                }
                                const int64_t key_pos = col_min + b;
                                const int64_t page_idx = key_pos / page_size;
                                const int64_t page_id = block_ptr[batch * max_pages + page_idx];
                                const int64_t slot = key_pos - page_idx * page_size;
                                const int64_t v_idx = ((page_id * num_kv_heads + kv_head) * page_size + slot) * D + c;
                                res += std::exp(s_i[a * Bc + b] - new_max) * value_ptr[v_idx];
                            }
                            o_i[a * D + c] = old_scale * o_i[a * D + c] + res;
                        }

                        l_i[a] = old_scale * l_i[a] + rowsum;
                        m_i[a] = new_max;
                    }
                }

                for (int64_t a = 0; a < br_upper_bound; a++) {
                    for (int64_t c = 0; c < D; c++) {
                        const int64_t out_idx = n * L * D + (i * Br + a) * D + c;
                        out_ptr[out_idx] = l_i[a] > 0.0f ? o_i[a * D + c] / l_i[a] : 0.0f;
                    }
                }
            }
        }
    });
}

#ifdef _METAL_
void PagedAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &q = inputs[0];
    const auto &key_pages = inputs[1];
    const auto &value_pages = inputs[2];
    const auto &block_table = inputs[3];
    const auto &context_lens = inputs[4];
    auto &out = outputs[0];

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("paged_attention: output dtype must be float32");
    }
    if (!q.flags().row_contiguous || !key_pages.flags().row_contiguous || !value_pages.flags().row_contiguous ||
        !block_table.flags().row_contiguous || !context_lens.flags().row_contiguous) {
        throw std::runtime_error("paged_attention: all inputs must be contiguous");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    auto library = d.get_library("tiny_llm_ext_ref");
    auto kernel = d.get_kernel("paged_attention_f32", library);

    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(key_pages, 1);
    compute_encoder.set_input_array(value_pages, 2);
    compute_encoder.set_input_array(block_table, 3);
    compute_encoder.set_input_array(context_lens, 4);
    compute_encoder.set_output_array(out, 5);

    const int N = q.shape()[0];
    const int L = q.shape()[1];
    const int D = q.shape()[2];
    const int page_size = key_pages.shape()[2];
    const int max_pages = block_table.shape()[1];
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(L, 7);
    compute_encoder.set_bytes(D, 8);
    compute_encoder.set_bytes(page_size, 9);
    compute_encoder.set_bytes(max_pages, 10);
    compute_encoder.set_bytes(static_cast<int>(is_causal_), 11);
    compute_encoder.set_bytes(num_kv_heads_, 12);
    compute_encoder.set_bytes(num_heads_, 13);
    compute_encoder.set_bytes(scale_, 14);

    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    size_t simd_width = kernel->threadExecutionWidth();

    const int Br = 32;
    const int Bc = 32;
    if (simd_width * Br > tgp_size) {
        throw std::runtime_error("paged_attention: simd_width * Br must fit in the threadgroup");
    }
    if (Bc > simd_width) {
        throw std::runtime_error("paged_attention: Bc must be less than or equal to simd_width");
    }
    if (D > 128) {
        throw std::runtime_error("paged_attention: head dimension must be less than or equal to 128");
    }

    const int Tr = (L + Br - 1) / Br;
    const int Tc = (max_pages * page_size + Bc - 1) / Bc;

    compute_encoder.set_bytes(Br, 15);
    compute_encoder.set_bytes(Bc, 16);
    compute_encoder.set_bytes(Tr, 17);
    compute_encoder.set_bytes(Tc, 18);

    compute_encoder.dispatch_threadgroups(MTL::Size(N, Tr, 1), MTL::Size(Br, simd_width, 1));
}
#else
void PagedAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("PagedAttention has no GPU implementation.");
}
#endif

}  // namespace tiny_llm_ext_ref
