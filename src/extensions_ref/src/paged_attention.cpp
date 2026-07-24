#include <algorithm>
#include <stdexcept>
#include <string>

#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext_ref {

mx::array paged_cache_update(const mx::array &pages, const mx::array &values, int page_id, int start,
                             mx::StreamOrDevice s) {
    if ((pages.dtype() != mx::float32 && pages.dtype() != mx::bfloat16) || values.dtype() != pages.dtype()) {
        throw std::runtime_error("paged_cache_update: pages and values must have the same float32 or bfloat16 dtype");
    }
    if (pages.shape().size() != 4 || values.shape().size() != 4 || values.shape()[0] != 1) {
        throw std::runtime_error(
            "paged_cache_update: expected pages [P, H, page_size, D] and values [1, H, length, D]");
    }
    if (values.shape()[1] != pages.shape()[1] || values.shape()[3] != pages.shape()[3]) {
        throw std::runtime_error("paged_cache_update: values must match the page head count and head dimension");
    }
    if (page_id < 0 || page_id >= pages.shape()[0] || start < 0 || start + values.shape()[2] > pages.shape()[2]) {
        throw std::runtime_error("paged_cache_update: destination slice is outside page storage");
    }
    return mx::array(pages.shape(), pages.dtype(), std::make_shared<PagedCacheUpdate>(to_stream(s), page_id, start),
                     {pages, values});
}

void PagedCacheUpdate::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("paged_cache_update: the course extension is GPU-only");
}

#ifdef _METAL_
void PagedCacheUpdate::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &pages = inputs[0];
    const auto &values = inputs[1];
    auto &out = outputs[0];
    if (!pages.flags().row_contiguous || !values.flags().row_contiguous) {
        throw std::runtime_error("paged_cache_update: pages and values must be contiguous");
    }

    // Page storage is request state, so the output intentionally aliases the
    // input buffer. The kernel touches only the appended slice instead of
    // building a functional copy of the entire cache tensor.
    out.copy_shared_buffer(pages);
    auto &d = mx::metal::device(stream().device);
    auto library = d.get_library("tiny_llm_ext_ref");
    const char *kernel_name = pages.dtype() == mx::bfloat16 ? "paged_cache_update_bf16" : "paged_cache_update_f32";
    auto kernel = d.get_kernel(kernel_name, library);
    auto &compute_encoder = mx::metal::get_command_encoder(stream());
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(values, 0);
    compute_encoder.set_output_array(out, 1);
    const int heads = pages.shape()[1];
    const int length = values.shape()[2];
    const int head_dim = pages.shape()[3];
    const int page_size = pages.shape()[2];
    compute_encoder.set_bytes(heads, 2);
    compute_encoder.set_bytes(length, 3);
    compute_encoder.set_bytes(head_dim, 4);
    compute_encoder.set_bytes(page_size, 5);
    compute_encoder.set_bytes(page_id_, 6);
    compute_encoder.set_bytes(start_, 7);
    const int threads = std::min<int>(kernel->maxTotalThreadsPerThreadgroup(), 256);
    compute_encoder.dispatch_threads(MTL::Size(values.size(), 1, 1), MTL::Size(threads, 1, 1));
}
#else
void PagedCacheUpdate::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("PagedCacheUpdate has no GPU implementation.");
}
#endif

mx::array paged_attention(const mx::array &q, const mx::array &key_pages, const mx::array &value_pages,
                          const mx::array &block_table, const mx::array &context_lens, const float scale,
                          const bool is_causal, const int num_kv_heads, const int num_heads, mx::StreamOrDevice s) {
    if ((q.dtype() != mx::float32 && q.dtype() != mx::bfloat16) || key_pages.dtype() != q.dtype() ||
        value_pages.dtype() != q.dtype()) {
        throw std::runtime_error(
            "paged_attention: q, key_pages, and value_pages must have the same float32 or bfloat16 dtype");
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

    return mx::array(q.shape(), q.dtype(),
                     std::make_shared<PagedAttention>(to_stream(s), scale, is_causal, num_kv_heads, num_heads),
                     {q, key_pages, value_pages, block_table, context_lens});
}

void PagedAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("paged_attention: the course extension is GPU-only");
}

#ifdef _METAL_
void PagedAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &q = inputs[0];
    const auto &key_pages = inputs[1];
    const auto &value_pages = inputs[2];
    const auto &block_table = inputs[3];
    const auto &context_lens = inputs[4];
    auto &out = outputs[0];

    if (!q.flags().row_contiguous || !key_pages.flags().row_contiguous || !value_pages.flags().row_contiguous ||
        !block_table.flags().row_contiguous || !context_lens.flags().row_contiguous) {
        throw std::runtime_error("paged_attention: all inputs must be contiguous");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    auto library = d.get_library("tiny_llm_ext_ref");
    auto &compute_encoder = mx::metal::get_command_encoder(s);

    const int N = q.shape()[0];
    const int L = q.shape()[1];
    const int D = q.shape()[2];
    const int page_size = key_pages.shape()[2];
    const int max_pages = block_table.shape()[1];
    const int is_causal = static_cast<int>(is_causal_);
    if (D <= 0 || D > 128) {
        throw std::runtime_error("paged_attention: head dimension must be in the range [1, 128]");
    }

    auto bind_arrays = [&]() {
        compute_encoder.set_input_array(q, 0);
        compute_encoder.set_input_array(key_pages, 1);
        compute_encoder.set_input_array(value_pages, 2);
        compute_encoder.set_input_array(block_table, 3);
        compute_encoder.set_input_array(context_lens, 4);
        compute_encoder.set_output_array(out, 5);
    };

    if (L <= 8) {
        const char *suffix =
            q.dtype() == mx::bfloat16 && D == 128 ? "bf16_d128" : (q.dtype() == mx::bfloat16 ? "bf16" : "f32");
        auto kernel = d.get_kernel(std::string("paged_attention_decode_") + suffix, library);
        compute_encoder.set_compute_pipeline_state(kernel);
        bind_arrays();
        compute_encoder.set_bytes(N, 6);
        compute_encoder.set_bytes(L, 7);
        compute_encoder.set_bytes(D, 8);
        compute_encoder.set_bytes(page_size, 9);
        compute_encoder.set_bytes(max_pages, 10);
        compute_encoder.set_bytes(is_causal, 11);
        compute_encoder.set_bytes(num_kv_heads_, 12);
        compute_encoder.set_bytes(num_heads_, 13);
        compute_encoder.set_bytes(scale_, 14);
        constexpr int simdgroups_per_query = 32;
        compute_encoder.set_threadgroup_memory_length(
            (simdgroups_per_query * 32 + 2 * simdgroups_per_query) * sizeof(float), 0);
        compute_encoder.dispatch_threadgroups(MTL::Size(N * L, 1, 1), MTL::Size(simdgroups_per_query * 32, 1, 1));
        return;
    }

    if (q.dtype() == mx::bfloat16) {
        if (D != 128) {
            throw std::runtime_error("paged_attention: bfloat16 prefill requires head dimension 128");
        }
        auto kernel = d.get_kernel("paged_attention_mma_bf16_d128", library);
        compute_encoder.set_compute_pipeline_state(kernel);
        bind_arrays();
        compute_encoder.set_bytes(N, 6);
        compute_encoder.set_bytes(L, 7);
        compute_encoder.set_bytes(page_size, 8);
        compute_encoder.set_bytes(max_pages, 9);
        compute_encoder.set_bytes(is_causal, 10);
        compute_encoder.set_bytes(num_kv_heads_, 11);
        compute_encoder.set_bytes(num_heads_, 12);
        compute_encoder.set_bytes(scale_, 13);
        const int batch_size = N / num_heads_;
        const int query_blocks = (L + 63) / 64;
        compute_encoder.dispatch_threadgroups(MTL::Size(query_blocks, num_heads_, batch_size), MTL::Size(32, 8, 1));
        return;
    }

    auto kernel = d.get_kernel("paged_attention_scalar_f32", library);
    compute_encoder.set_compute_pipeline_state(kernel);
    bind_arrays();
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(L, 7);
    compute_encoder.set_bytes(D, 8);
    compute_encoder.set_bytes(page_size, 9);
    compute_encoder.set_bytes(max_pages, 10);
    compute_encoder.set_bytes(is_causal, 11);
    compute_encoder.set_bytes(num_kv_heads_, 12);
    compute_encoder.set_bytes(num_heads_, 13);
    compute_encoder.set_bytes(scale_, 14);
    const int query_blocks = (L + 15) / 16;
    compute_encoder.dispatch_threadgroups(MTL::Size(N, query_blocks, 1), MTL::Size(32, 16, 1));
}
#else
void PagedAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("PagedAttention has no GPU implementation.");
}
#endif

}  // namespace tiny_llm_ext_ref
