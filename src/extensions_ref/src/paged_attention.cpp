#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/transforms.h"
#include "mlx/version.h"
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

#ifdef _METAL_
namespace {

constexpr int grouped_width = 5;
constexpr int grouped_heads = 8;
constexpr int grouped_page_size = 128;
constexpr int grouped_head_dim = 128;
constexpr int record_size = 4;

bool compact_binding(const mx::array &array) {
    if (!array.data_shared_ptr() || !array.data_shared_ptr()->is_owned() ||
        !array.flags().row_contiguous || array.data_size() != array.size() || array.offset() < 0 ||
        array.offset() % array.itemsize() != 0 || !array.buffer().ptr()) {
        return false;
    }
    const auto capacity = array.buffer_size();
    const auto offset = static_cast<size_t>(array.offset());
    return offset <= capacity && array.nbytes() <= capacity - offset;
}

bool separate_projection(const mx::array &key_pages, const mx::array &value_pages, const mx::array &projected) {
    return compact_binding(projected) && projected.buffer().ptr() != key_pages.buffer().ptr() &&
           projected.buffer().ptr() != value_pages.buffer().ptr();
}

bool valid_tail_records(const std::vector<int32_t> &records, int capacity) {
    if (records.size() < 2 * record_size || records.size() > grouped_width * record_size ||
        records.size() % record_size != 0) {
        return false;
    }
    int previous_slot = -1;
    for (size_t i = 0; i < records.size(); i += record_size) {
        const int slot = records[i], page = records[i + 1], tail = records[i + 2];
        if (slot <= previous_slot || slot >= grouped_width || page < 0 || page >= capacity ||
            tail <= 0 || tail >= grouped_page_size) {
            return false;
        }
        for (size_t j = 1; j < i; j += record_size) {
            if (records[j] == page) {
                return false;
            }
        }
        previous_slot = slot;
    }
    return true;
}

class GroupedPagedCacheUpdate final : public mx::Primitive {
public:
    GroupedPagedCacheUpdate(mx::Stream stream, const std::vector<int32_t> &records)
        : mx::Primitive(stream), records_(records) {}

    const char *name() const override { return "GroupedPagedCacheUpdate"; }

    void eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) override {
        throw std::runtime_error("Grouped paged cache update requires Metal");
    }

    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override {
        auto &out = outputs[0];
        out.copy_shared_buffer(inputs[0]);
        auto &device = mx::metal::device(stream().device);
        auto *library = device.get_library("tiny_llm_ext_ref");
        auto *kernel = device.get_kernel("paged_cache_update_grouped_bf16", library);
        auto &encoder = mx::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(inputs[1], 0);
        encoder.set_output_array(out, 1);
        encoder.set_vector_bytes(records_, 2);
        const int threads = std::min<int>(kernel->maxTotalThreadsPerThreadgroup(), 256);
        encoder.dispatch_threads(MTL::Size(grouped_heads * grouped_head_dim, records_.size() / record_size, 1),
                                 MTL::Size(threads, 1, 1));
    }

private:
    std::vector<int32_t> records_;
};

}  // namespace
#endif

std::optional<std::vector<mx::array>> try_grouped_paged_cache_update(
    const mx::array &key_pages, const mx::array &value_pages, const mx::array &keys,
    const mx::array &values, const std::vector<int32_t> &records) {
#ifdef _METAL_
    const auto stream = mx::to_stream(mx::StreamOrDevice{});
    if (stream.device.type != mx::Device::gpu || !mx::metal::is_available() ||
        std::string_view(mx::version()) != "0.32.4" ||
        key_pages.ndim() != 4 || key_pages.shape() != value_pages.shape() ||
        key_pages.shape(1) != grouped_heads || key_pages.shape(2) != grouped_page_size ||
        key_pages.shape(3) != grouped_head_dim || key_pages.size() > std::numeric_limits<int32_t>::max() ||
        keys.shape() != mx::Shape{grouped_width, grouped_heads, 1, grouped_head_dim} || values.shape() != keys.shape() ||
        key_pages.dtype() != mx::bfloat16 || value_pages.dtype() != mx::bfloat16 ||
        keys.dtype() != mx::bfloat16 || values.dtype() != mx::bfloat16 ||
        key_pages.is_tracer() || value_pages.is_tracer() || keys.is_tracer() || values.is_tracer() ||
        !valid_tail_records(records, key_pages.shape(0))) {
        return std::nullopt;
    }
    const std::string_view architecture = mx::metal::device(stream.device).get_architecture();
    if (architecture != "applegpu_g16s" ||
        key_pages.status() == mx::array::Status::unscheduled || value_pages.status() == mx::array::Status::unscheduled ||
        !compact_binding(key_pages) || !compact_binding(value_pages) ||
        key_pages.buffer().ptr() == value_pages.buffer().ptr()) {
        return std::nullopt;
    }
    if ((keys.status() != mx::array::Status::unscheduled && !separate_projection(key_pages, value_pages, keys)) ||
        (values.status() != mx::array::Status::unscheduled && !separate_projection(key_pages, value_pages, values))) {
        return std::nullopt;
    }

    // Lazy views can change strides or share the backing only when encoded.
    // Resolve both descriptors before constructing either mutating writer.
    mx::async_eval(keys, values);
    if (!separate_projection(key_pages, value_pages, keys) || !separate_projection(key_pages, value_pages, values)) {
        return std::nullopt;
    }

    auto primitive = std::make_shared<GroupedPagedCacheUpdate>(stream, records);
    auto key_root = mx::array(key_pages.shape(), key_pages.dtype(), primitive, {key_pages, keys});
    auto value_root = mx::array(value_pages.shape(), value_pages.dtype(), std::move(primitive), {value_pages, values});
    return std::vector<mx::array>{std::move(key_root), std::move(value_root)};
#else
    (void)key_pages;
    (void)value_pages;
    (void)keys;
    (void)values;
    (void)records;
    return std::nullopt;
#endif
}

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

    if (q.dtype() == mx::bfloat16 && D == 128) {
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

    const char *kernel_name = q.dtype() == mx::bfloat16 ? "paged_attention_scalar_bf16" : "paged_attention_scalar_f32";
    auto kernel = d.get_kernel(kernel_name, library);
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
