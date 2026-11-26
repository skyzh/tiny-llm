#include "week2_bf16_history.h"

#include <stdexcept>

#ifdef TINY_LLM_WEEK2_BF16_HISTORY
#include <cstdint>
#include <string_view>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/version.h"
#endif

namespace tiny_llm_ext_ref {
namespace mx = mlx::core;

#ifdef TINY_LLM_WEEK2_BF16_HISTORY
namespace {

constexpr int head_dim = 128;
constexpr int kv_heads = 8;
constexpr int queries_per_head = 4;
constexpr int query_heads = kv_heads * queries_per_head;

bool valid_history_shape(const mx::array &history) {
    return history.ndim() == 4 && (history.shape(2) == 4092 || history.shape(2) == 4096) &&
           history.shape() == mx::Shape{1, kv_heads, history.shape(2), head_dim} &&
           history.dtype() == mx::bfloat16 && !history.is_tracer();
}

bool can_bind_history(const mx::array &history) {
    const int64_t pitch = history.strides(1);
    const int64_t row_span = int64_t(history.shape(2)) * head_dim;
    if (!history.data_shared_ptr() || history.offset() != 0 || history.strides(3) != 1 ||
        history.strides(2) != head_dim || pitch < row_span || pitch % head_dim != 0) {
        return false;
    }
    // The last addressed row must fit the actual allocation, including head padding.
    const auto capacity = history.buffer_size() / history.itemsize();
    return capacity >= row_span && uint64_t(pitch) <= (capacity - row_span) / (kv_heads - 1);
}

mx::Shape grouped_row_shape(int width) {
    return {1, 1, kv_heads, queries_per_head, 1, width};
}

class HistoryContraction final : public mx::UnaryPrimitive {
public:
    HistoryContraction(mx::Stream stream, bool score)
        : mx::UnaryPrimitive(stream), score_(score) {}

    const char *name() const override { return "Week2BF16HistoryContraction"; }

    void eval_cpu(const std::vector<mx::array> &, mx::array &) override {
        throw std::runtime_error("Week 2 BF16 history contractions require Metal");
    }

    void eval_gpu(const std::vector<mx::array> &inputs, mx::array &out) override {
        const auto &history = inputs[0];
        const auto &rows = inputs[1];
        const int context = history.shape(2);
        const int reduction = score_ ? head_dim : context;
        const int columns = score_ ? context : head_dim;

        auto &device = mx::metal::device(stream().device);
        auto *library = device.get_library("tiny_llm_ext_ref");
        auto *kernel = device.get_kernel(
            score_ ? "tiny_llm_week2_history_scores" : "tiny_llm_week2_history_values", library);
        const std::vector<int> batch_shape{kv_heads, queries_per_head};
        const std::vector<int64_t> row_pitch{rows.strides(2), rows.strides(3)};
        const std::vector<int64_t> history_pitch{history.strides(1), 0};

        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto &encoder = mx::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(history, 0);
        encoder.set_input_array(rows, 1);
        encoder.set_output_array(out, 3);
        encoder.set_bytes(reduction, 4);
        encoder.set_bytes(columns, 5);
        encoder.set_bytes(head_dim, 6);
        encoder.set_bytes(int(batch_shape.size()), 9);
        encoder.set_vector_bytes(batch_shape, 10);
        encoder.set_vector_bytes(row_pitch, 11);
        encoder.set_vector_bytes(history_pitch, 12);
        // QK uses the four-row GEMV tile. PV uses transposed GEMV with 32 columns per group.
        if (score_) {
            encoder.dispatch_threadgroups(MTL::Size((context + 15) / 16, 1, query_heads), MTL::Size(32, 1, 4));
        } else {
            encoder.dispatch_threadgroups(MTL::Size(head_dim / 32, 1, query_heads), MTL::Size(32, 2, 1));
        }
    }

private:
    bool score_;
};
}  // namespace
#endif

std::optional<mx::array> try_week2_bf16_history_scores(
    const mx::array &query, const mx::array &key, const mx::array &value,
    const std::optional<mx::array> &mask) {
#ifdef TINY_LLM_WEEK2_BF16_HISTORY
    const auto stream = mx::to_stream(mx::StreamOrDevice{});
    if (stream.device.type != mx::Device::gpu || !mx::metal::is_available() ||
        std::string_view(mx::version()) != "0.32.4" ||
        query.shape() != mx::Shape{1, query_heads, 1, head_dim} ||
        query.dtype() != mx::bfloat16 || query.is_tracer() ||
        !valid_history_shape(key) || !valid_history_shape(value) || key.shape() != value.shape() ||
        (mask && mask->is_tracer())) {
        return std::nullopt;
    }
    auto &device = mx::metal::device(stream.device);
    const std::string_view architecture = device.get_architecture();
    if (device.get_architecture_gen() != 16 || architecture.empty() || architecture.back() != 's' ||
        architecture != device.mtl_device()->architecture()->name()->utf8String()) {
        return std::nullopt;
    }
    // Cache slices are lazy. Check both reached BF16 views before choosing either loader.
    mx::eval(query, key, value);
    if (!query.flags().row_contiguous || query.offset() != 0 ||
        !can_bind_history(key) || !can_bind_history(value)) {
        return std::nullopt;
    }
    const auto rows = mx::reshape(mx::astype(query, mx::float32, stream),
                                  grouped_row_shape(head_dim), stream);
    return mx::array(grouped_row_shape(key.shape(2)), mx::float32,
                     std::make_shared<HistoryContraction>(stream, true), {key, rows});
#else
    (void)query;
    (void)key;
    (void)value;
    (void)mask;
    return std::nullopt;
#endif
}

mx::array week2_bf16_history_values(const mx::array &value, const mx::array &weights) {
#ifdef TINY_LLM_WEEK2_BF16_HISTORY
    const auto stream = mx::to_stream(mx::StreamOrDevice{});
    if (stream.device.type == mx::Device::gpu && valid_history_shape(value) &&
        value.is_available() && can_bind_history(value) && weights.dtype() == mx::float32 &&
        weights.shape() == grouped_row_shape(value.shape(2)) && !weights.is_tracer()) {
        auto rows = mx::contiguous(weights, false, stream);
        return mx::array(grouped_row_shape(head_dim), mx::float32,
                         std::make_shared<HistoryContraction>(stream, false), {value, rows});
    }
#else
    (void)value;
    (void)weights;
#endif
    throw std::invalid_argument("Week 2 BF16 history product requires an admitted value view and FP32 weights");
}

}  // namespace tiny_llm_ext_ref
