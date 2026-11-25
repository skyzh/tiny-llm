#include "tiny_llm_ext.h"

#ifdef TINY_LLM_WEEK1_SAMPLED_HEAD
#include <cstdint>
#include <string_view>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/transforms.h"
#include "mlx/version.h"
#endif

namespace tiny_llm_ext_ref {
#ifdef TINY_LLM_WEEK1_SAMPLED_HEAD
namespace {

constexpr int rows = 128;
constexpr int width = 2560;
constexpr int vocabulary = 151936;
constexpr int tile_rows = 64;
constexpr auto entry = "tiny_llm_week1_sampled_head";

bool dense_base(const mx::array &x) {
    return x.flags().row_contiguous && x.strides(-1) == 1 && x.strides(-2) == width &&
           x.data_shared_ptr() && x.offset() == 0 && x.nbytes() <= x.buffer_size();
}

class SampledHead : public mx::UnaryPrimitive {
public:
    explicit SampledHead(mx::Stream stream) : mx::UnaryPrimitive(stream) {}

    const char *name() const override { return "Week1SampledHead"; }

    void eval_cpu(const std::vector<mx::array> &, mx::array &) override {
        throw std::runtime_error("Week 1 sampled head requires a Metal stream");
    }

    void eval_gpu(const std::vector<mx::array> &inputs, mx::array &out) override {
        constexpr int tiles_n = vocabulary / 64;
        constexpr bool disabled = false;
        constexpr bool aligned = true;
        const mx::metal::MTLFCList constants{
            {&disabled, MTL::DataType::DataTypeBool, 10},
            {&disabled, MTL::DataType::DataTypeBool, 100},
            {&disabled, MTL::DataType::DataTypeBool, 110},
            {&aligned, MTL::DataType::DataTypeBool, 200},
            {&aligned, MTL::DataType::DataTypeBool, 201},
            {&aligned, MTL::DataType::DataTypeBool, 202},
        };
        auto &device = mx::metal::device(stream().device);
        auto *library = device.get_library("tiny_llm_ext_ref");
        auto *kernel = device.get_kernel(entry, library, entry, constants);
        const mlx::steel::GEMMParams params{
            tile_rows, vocabulary, width, width, width, vocabulary, tiles_n, 1,
            int64_t(tile_rows) * width, 0, int64_t(tile_rows) * vocabulary,
            0, width / 16, 1,
        };
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto &encoder = mx::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        // Offset into the original normalized backing. Row 127 becomes row 63.
        encoder.set_input_array(inputs[0], 0, int64_t(rows - tile_rows) * width * inputs[0].itemsize());
        encoder.set_input_array(inputs[1], 1);
        encoder.set_output_array(out, 3);
        encoder.set_bytes(params, 4);
        encoder.dispatch_threadgroups(MTL::Size(tiles_n, 1, 1), MTL::Size(32, 2, 2));
    }
};
}  // namespace
#endif

std::optional<mx::array> try_week1_sampled_head(const mx::array &normalized, const mx::array &weight) {
#ifdef TINY_LLM_WEEK1_SAMPLED_HEAD
    const auto stream = mx::to_stream(mx::StreamOrDevice{});
    if (stream.device.type != mx::Device::gpu || !mx::metal::is_available() ||
        std::string_view(mx::version()) != "0.32.4" || normalized.shape() != mx::Shape{1, rows, width} ||
        weight.shape() != mx::Shape{vocabulary, width} || normalized.dtype() != mx::bfloat16 ||
        weight.dtype() != mx::bfloat16 || normalized.is_tracer() || weight.is_tracer()) {
        return std::nullopt;
    }

    // This MLX version selects the ordinary NT 64x64x16 Steel entry for this
    // geometry on the tested M4 Pro GPU. Other GPU families use the stock path.
    auto &device = mx::metal::device(stream.device);
    const std::string_view architecture = device.get_architecture();
    if (device.get_architecture_gen() != 16 || architecture.empty() || architecture.back() != 's' ||
        architecture != device.mtl_device()->architecture()->name()->utf8String()) {
        return std::nullopt;
    }
    // Lazy arrays can acquire view offsets and different strides at evaluation.
    // Inspect the reached layout before bypassing the stock matmul dispatcher.
    mx::eval(normalized, weight);
    if (!dense_base(normalized) || !dense_base(weight)) {
        return std::nullopt;
    }
    return mx::array({1, 1, vocabulary}, mx::bfloat16,
                     std::make_shared<SampledHead>(stream), {normalized, weight});
#else
    (void)normalized;
    (void)weight;
    return std::nullopt;
#endif
}

}  // namespace tiny_llm_ext_ref
