#include <cstdint>

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext_ref {

mx::array quantized_matmul(const mx::array &scales,  // Input array scales
                           const mx::array &biases,  // Input array biases
                           const int group_size,     // Group size
                           const int bits,           // Number of bits
                           const mx::array &a,       // Input array a (not quantized)
                           const mx::array &b,       // Input array b (quantized)
                           const bool transpose_b,   // Whether to transpose b
                           const bool use_simdgroup,
                           mx::StreamOrDevice s /* = {} */  // Stream on which to schedule the operation
) {
    if (scales.dtype() != mx::float16 && scales.dtype() != mx::bfloat16) {
        throw std::runtime_error("quantized_matmul: scales must be float16 or bfloat16");
    }
    if (scales.dtype() != biases.dtype()) {
        throw std::runtime_error("quantized_matmul: scales and biases must be the same dtype");
    }
    if (b.dtype() != mx::uint32) {
        throw std::runtime_error("quantized_matmul: b must be uint32");
    }
    if (a.dtype() != scales.dtype()) {
        throw std::runtime_error("quantized_matmul: a must be the same dtype as scales");
    }
    if (a.shape().size() != 2) {
        throw std::runtime_error("quantized_matmul: a must be a 2D array");
    }
    if (b.shape().size() != 2) {
        throw std::runtime_error("quantized_matmul: b must be a 2D array");
    }
    if (bits != 4) {
        throw std::runtime_error("quantized_matmul: bits must be 4");
    }
    const int packs_per_item = 32 / bits;
    if (group_size != 128) {
        throw std::runtime_error("quantized_matmul: group_size must be 128");
    }
    auto out_shape = a.shape();
    if (out_shape.size() != 2) {
        throw std::runtime_error("quantized_matmul: a must be a 2D array");
    }
    out_shape[1] = b.shape()[0];
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: b must be transposed");
    }

    if (scales.shape() != biases.shape()) {
        throw std::runtime_error("quantized_matmul: scales and biases must have the same shape");
    }
    if (b.shape()[0] != scales.shape()[0]) {
        throw std::runtime_error("quantized_matmul: b must have the same number of rows as scales");
    }
    if (a.shape()[1] % group_size != 0) {
        throw std::runtime_error("quantized_matmul: a columns must be divisible by group_size");
    }
    if (scales.shape()[1] != a.shape()[1] / group_size) {
        throw std::runtime_error("quantized_matmul: scales must have one column per input group");
    }
    if (b.shape()[1] != a.shape()[1] / packs_per_item) {
        throw std::runtime_error("quantized_matmul: a must have the same number of columns as b");
    }

    return mx::array(
        /* const mx::Shape& shape = */ out_shape,
        /* mx::Dtype dtype = */ a.dtype(),
        /* std::shared_ptr<mx::Primitive> primitive = */
        std::make_shared<QuantizedMatmul>(to_stream(s), use_simdgroup),
        /* const std::vector<mx::array>& inputs = */ {scales, biases, a, b});
}

mx::array quantized_embedding(const mx::array &indices, const mx::array &scales, const mx::array &biases,
                              const mx::array &weight, int group_size, int bits, mx::StreamOrDevice s) {
    if ((indices.dtype() != mx::int32 && indices.dtype() != mx::uint32) || weight.dtype() != mx::uint32) {
        throw std::runtime_error("quantized_embedding: indices and weight must use 32-bit integers");
    }
    if (scales.dtype() != biases.dtype() || (scales.dtype() != mx::float16 && scales.dtype() != mx::bfloat16)) {
        throw std::runtime_error("quantized_embedding: scales and biases must have the same 16-bit dtype");
    }
    if (group_size != 128 || bits != 4 || scales.shape() != biases.shape()) {
        throw std::runtime_error("quantized_embedding: expected 4-bit weights with group size 128");
    }
    const int dim = weight.shape()[1] * (32 / bits);
    if (scales.shape()[0] != weight.shape()[0] || scales.shape()[1] != dim / group_size) {
        throw std::runtime_error("quantized_embedding: incompatible parameter shapes");
    }
    auto out_shape = indices.shape();
    out_shape.push_back(dim);
    return mx::array(out_shape, scales.dtype(), std::make_shared<QuantizedEmbedding>(to_stream(s)),
                     {indices, scales, biases, weight});
}

template <typename T>
void quantized_matmul_impl(const mx::array &scales, const mx::array &biases, const mx::array &a, const mx::array &b,
                           mx::array &out, mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    if (!a.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: a must be contiguous");
    }
    if (!b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: b must be contiguous");
    }

    encoder.dispatch([out_ptr = out.data<T>(), out_shape = out.shape(), out_strides = out.strides(),
                      a = mx::array::unsafe_weak_copy(a), b = mx::array::unsafe_weak_copy(b),
                      scales = mx::array::unsafe_weak_copy(scales), biases = mx::array::unsafe_weak_copy(biases)]() {
        int M = a.shape()[0];
        int N = a.shape()[1];
        int K = b.shape()[0];
        const int group_size = 128;
        const int bits = 4;
        const int group_per_row = N / group_size;
        const T *a_ptr = a.data<T>();
        const uint32_t *b_ptr = b.data<uint32_t>();
        const T *scales_ptr = scales.data<T>();
        const T *biases_ptr = biases.data<T>();
        uint32_t item_mask = (1 << bits) - 1;
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float sum = 0;
                for (int group_idx = 0; group_idx < group_per_row; group_idx++) {
                    int64_t scales_loc =
                        mx::elem_to_loc(k * group_per_row + group_idx, scales.shape(), scales.strides());
                    int64_t biases_loc =
                        mx::elem_to_loc(k * group_per_row + group_idx, biases.shape(), biases.strides());
                    T scale = scales_ptr[scales_loc];
                    T bias = biases_ptr[biases_loc];
                    int64_t b_loc = mx::elem_to_loc((k * N + group_idx * group_size) / 8, b.shape(), b.strides());
                    int64_t a_loc = mx::elem_to_loc(i * N + group_idx * group_size, a.shape(), a.strides());
                    const int packs_per_item = 32 / bits;
                    for (int item_idx = 0; item_idx < group_size; item_idx += packs_per_item) {
                        uint32_t b_val = b_ptr[b_loc];
                        uint8_t *b_bytes = reinterpret_cast<uint8_t *>(&b_val);
                        for (int pack_idx = 0; pack_idx < packs_per_item; pack_idx++) {
                            uint8_t item_val = (b_bytes[pack_idx / 2] >> ((pack_idx % 2) * bits)) & item_mask;
                            float b =
                                static_cast<float>(item_val) * static_cast<float>(scale) + static_cast<float>(bias);
                            float a = static_cast<float>(a_ptr[a_loc]);
                            sum += a * b;
                            a_loc += 1;
                        }
                        b_loc += 1;
                    }
                }
                int64_t out_loc = mx::elem_to_loc(i * K + k, out_shape, out_strides);
                out_ptr[out_loc] = static_cast<T>(sum);
            }
        }
    });
}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    if (out.dtype() == mx::float16) {
        return quantized_matmul_impl<mx::float16_t>(scales, biases, a, b, out, stream());
    } else if (out.dtype() == mx::bfloat16) {
        return quantized_matmul_impl<mx::bfloat16_t>(scales, biases, a, b, out, stream());
    } else {
        throw std::runtime_error("quantized_matmul: output must be float16 or bfloat16");
    }
}

template <typename T, typename IndexT>
void quantized_embedding_cpu_impl(const std::vector<mx::array> &inputs, mx::array &out, mx::Stream stream) {
    const auto &indices = inputs[0];
    const auto &scales = inputs[1];
    const auto &biases = inputs[2];
    const auto &weight = inputs[3];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &encoder = mx::cpu::get_command_encoder(stream);
    for (const auto &input : inputs) {
        encoder.set_input_array(input);
    }
    encoder.set_output_array(out);
    encoder.dispatch([indices = mx::array::unsafe_weak_copy(indices), scales = mx::array::unsafe_weak_copy(scales),
                      biases = mx::array::unsafe_weak_copy(biases), weight = mx::array::unsafe_weak_copy(weight),
                      out_ptr = out.data<T>(), size = out.size()]() {
        constexpr int group_size = 128;
        constexpr int values_per_word = 8;
        const int dim = weight.shape()[1] * values_per_word;
        const int groups_per_row = dim / group_size;
        const IndexT *indices_ptr = indices.data<IndexT>();
        const uint32_t *weight_ptr = weight.data<uint32_t>();
        const T *scales_ptr = scales.data<T>();
        const T *biases_ptr = biases.data<T>();
        for (size_t index = 0; index < size; ++index) {
            const int token = index / dim;
            const int column = index % dim;
            const int row = indices_ptr[token];
            const uint32_t packed = weight_ptr[row * weight.shape()[1] + column / values_per_word];
            const float quantized = static_cast<float>((packed >> ((column % values_per_word) * 4)) & 15);
            const int parameter = row * groups_per_row + column / group_size;
            out_ptr[index] = static_cast<T>(quantized * static_cast<float>(scales_ptr[parameter]) +
                                            static_cast<float>(biases_ptr[parameter]));
        }
    });
}

void QuantizedEmbedding::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    if (outputs[0].dtype() == mx::float16 && inputs[0].dtype() == mx::int32) {
        quantized_embedding_cpu_impl<mx::float16_t, int32_t>(inputs, outputs[0], stream());
    } else if (outputs[0].dtype() == mx::float16) {
        quantized_embedding_cpu_impl<mx::float16_t, uint32_t>(inputs, outputs[0], stream());
    } else if (inputs[0].dtype() == mx::int32) {
        quantized_embedding_cpu_impl<mx::bfloat16_t, int32_t>(inputs, outputs[0], stream());
    } else {
        quantized_embedding_cpu_impl<mx::bfloat16_t, uint32_t>(inputs, outputs[0], stream());
    }
}

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // Make a kernel from this metal library
    auto library = d.get_library("tiny_llm_ext_ref");
    // The readable scalar kernel must remain independently benchmarkable at
    // every shape. Only select the optimized matvec when the caller requested
    // the optimized path.
    const bool use_matvec = use_simdgroup_ && a.shape()[0] <= 8;
    const char *kernel_name;
    if (use_matvec) {
        const bool use_wide_matvec = b.shape()[0] >= 8192;
        if (use_wide_matvec) {
            kernel_name = out.dtype() == mx::float16 ? "quantized_matvec_x8_w4a16_g128_f16"
                                                     : "quantized_matvec_x8_w4a16_g128_bf16";
        } else {
            kernel_name = out.dtype() == mx::float16 ? "quantized_matvec_x2_w4a16_g128_f16"
                                                     : "quantized_matvec_x2_w4a16_g128_bf16";
        }
    } else if (use_simdgroup_) {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matmul_simdgroup_w4a16_g128_f16"
                                                 : "quantized_matmul_simdgroup_w4a16_g128_bf16";
    } else {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matmul_vanilla_w4a16_g128_f16"
                                                 : "quantized_matmul_vanilla_w4a16_g128_bf16";
    }
    auto kernel = d.get_kernel(kernel_name, library);

    // Prepare to encode kernel
    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    // Encode input arrays to kernel
    compute_encoder.set_input_array(scales, 0);
    compute_encoder.set_input_array(biases, 1);
    compute_encoder.set_input_array(a, 2);
    compute_encoder.set_input_array(b, 3);
    // Encode output arrays to kernel
    compute_encoder.set_output_array(out, 4);

    if (!a.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: a must be contiguous");
    }
    if (!b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: b must be contiguous");
    }

    int M = a.shape()[0];
    int N = a.shape()[1];
    int K = b.shape()[0];

    if (N % 128 != 0) {
        throw std::runtime_error("quantized_matmul: N must be divisible by group_size");
    }

    // Encode matrix parameters
    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);

    if (use_matvec) {
        const int outputs_per_simdgroup = K >= 8192 ? 8 : 2;
        constexpr int simdgroups_per_threadgroup = 8;
        const int outputs_per_threadgroup = simdgroups_per_threadgroup * outputs_per_simdgroup;
        const int column_tiles = (K + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
        MTL::Size num_threadgroups = MTL::Size(M * column_tiles, 1, 1);
        MTL::Size num_threads_per_group = MTL::Size(simdgroups_per_threadgroup * 32, 1, 1);
        compute_encoder.dispatch_threadgroups(num_threadgroups, num_threads_per_group);
        return;
    }

    if (use_simdgroup_) {
        constexpr int tile_size = 8;
        constexpr int simdgroups_per_threadgroup = 8;
        const int row_tiles = (M + tile_size - 1) / tile_size;
        const int column_tiles = (K + tile_size - 1) / tile_size;
        const int threadgroups =
            (row_tiles * column_tiles + simdgroups_per_threadgroup - 1) / simdgroups_per_threadgroup;
        compute_encoder.dispatch_threadgroups(MTL::Size(threadgroups, 1, 1),
                                              MTL::Size(simdgroups_per_threadgroup * 32, 1, 1));
        return;
    }

    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    int x_size = 32;
    if (M <= 16) {
        x_size = 16;
    }
    const int y_size = tgp_size / x_size;
    if (tgp_size < x_size * y_size) {
        throw std::runtime_error("quantized_matmul: tgp_size must be larger than x*y");
    }
    MTL::Size num_threadgroups = MTL::Size((M + x_size - 1) / x_size, (K + y_size - 1) / y_size, 1);
    MTL::Size num_threads_per_group = MTL::Size(x_size, y_size, 1);

    // MTL::Size num_threadgroups = MTL::Size((M * K + tgp_size - 1) / tgp_size, 1, 1);
    // MTL::Size num_threads_per_group = MTL::Size(tgp_size, 1, 1);

    // Launch the grid with the given number of threads divided among
    // the given threadgroups
    compute_encoder.dispatch_threadgroups(num_threadgroups, num_threads_per_group);
}

void QuantizedEmbedding::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &indices = inputs[0];
    const auto &scales = inputs[1];
    const auto &biases = inputs[2];
    const auto &weight = inputs[3];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    const bool unsigned_indices = indices.dtype() == mx::uint32;
    const char *kernel_name;
    if (out.dtype() == mx::float16) {
        kernel_name =
            unsigned_indices ? "quantized_embedding_w4a16_g128_f16_u32" : "quantized_embedding_w4a16_g128_f16_i32";
    } else {
        kernel_name =
            unsigned_indices ? "quantized_embedding_w4a16_g128_bf16_u32" : "quantized_embedding_w4a16_g128_bf16_i32";
    }
    auto kernel = d.get_kernel(kernel_name, d.get_library("tiny_llm_ext_ref"));
    auto &encoder = d.get_command_encoder(stream().index);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(indices, 0);
    encoder.set_input_array(scales, 1);
    encoder.set_input_array(biases, 2);
    encoder.set_input_array(weight, 3);
    encoder.set_output_array(out, 4);
    const int tokens = indices.size();
    const int dim = out.shape().back();
    encoder.set_bytes(tokens, 5);
    encoder.set_bytes(dim, 6);
    const int threads = std::min<int>(kernel->maxTotalThreadsPerThreadgroup(), 256);
    encoder.dispatch_threads(MTL::Size(out.size(), 1, 1), MTL::Size(threads, 1, 1));
}

}  // namespace tiny_llm_ext_ref
