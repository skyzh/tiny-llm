#include <cstdint>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext_ref {

mx::array quantized_matmul(const mx::array &scales,         // Input array scales
                           const mx::array &biases,         // Input array biases
                           const int group_size,            // Group size
                           const int bits,                  // Number of bits
                           const mx::array &a,              // Input array a (not quantized)
                           const mx::array &b,              // Input array b (quantized)
                           const bool transpose_b,          // Whether to transpose b
                           mx::StreamOrDevice s /* = {} */  // Stream on which to schedule the operation
) {
    if (scales.dtype() != mx::float16 && scales.dtype() != mx::bfloat16 && scales.dtype() != mx::float32) {
        throw std::runtime_error("quantized_matmul: scales must be float16 or bfloat16 or float32");
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
    if (group_size != 64) {
        throw std::runtime_error("quantized_matmul: group_size must be 64");
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
    if (b.shape()[1] != scales.shape()[1] * group_size / 8) {
        throw std::runtime_error("quantized_matmul: a must have the same number of columns as scales");
    }
    if (a.shape()[1] != b.shape()[1] * 8) {
        throw std::runtime_error("quantized_matmul: a must have the same number of columns as b");
    }

    return mx::array(
        /* const mx::Shape& shape = */ out_shape,
        /* mx::Dtype dtype = */ a.dtype(),
        /* std::shared_ptr<mx::Primitive> primitive = */
        std::make_shared<QuantizedMatmul>(to_stream(s), group_size, bits),
        /* const std::vector<mx::array>& inputs = */ {scales, biases, a, b});
}

void quantized_matmul_impl(const mx::array &scales, const mx::array &biases, const mx::array &a, const mx::array &b,
                           mx::array &out, int group_size, int bits, mx::Stream stream) {
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

    // Launch the CPU kernel, TODO: support bfloat16
    encoder.dispatch([out_ptr = out.data<float16_t>(), out_shape = out.shape(), out_strides = out.strides(),
                      a = mx::array::unsafe_weak_copy(a), b = mx::array::unsafe_weak_copy(b),
                      scales = mx::array::unsafe_weak_copy(scales), biases = mx::array::unsafe_weak_copy(biases)]() {
        int M = a.shape()[0];
        int N = a.shape()[1];
        int K = b.shape()[0];
        const int group_size = 64;
        const int bits = 4;
        const int group_per_row = N / group_size;
        const float16_t *a_ptr = a.data<float16_t>();
        const uint32_t *b_ptr = b.data<uint32_t>();
        const float16_t *scales_ptr = scales.data<float16_t>();
        const float16_t *biases_ptr = biases.data<float16_t>();
        uint32_t item_mask = (1 << bits) - 1;
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float sum = 0;
                for (int group_idx = 0; group_idx < group_per_row; group_idx++) {
                    int64_t scales_loc =
                        mx::elem_to_loc(k * group_per_row + group_idx, scales.shape(), scales.strides());
                    int64_t biases_loc =
                        mx::elem_to_loc(k * group_per_row + group_idx, biases.shape(), biases.strides());
                    float16_t scale = scales_ptr[scales_loc];
                    float16_t bias = biases_ptr[biases_loc];
                    int64_t b_loc = mx::elem_to_loc((k * N + group_idx * group_size) / 8, b.shape(), b.strides());
                    int64_t a_loc = mx::elem_to_loc(i * N + group_idx * group_size, a.shape(), a.strides());
                    const int packs_per_item = 32 / bits;
                    for (int item_idx = 0; item_idx < group_size; item_idx += packs_per_item) {
                        uint32_t b_val = b_ptr[b_loc];
                        uint8_t *b_bytes = reinterpret_cast<uint8_t *>(&b_val);
                        for (int pack_idx = 0; pack_idx < packs_per_item; pack_idx++) {
                            uint8_t item_val = (b_bytes[pack_idx / 2] >> ((pack_idx % 2) * bits)) & item_mask;
                            float b = static_cast<float>(item_val) * scale + bias;
                            float a = a_ptr[a_loc];
                            sum += a * b;
                            a_loc += 1;
                        }
                        b_loc += 1;
                    }
                }
                int64_t out_loc = mx::elem_to_loc(i * K + k, out_shape, out_strides);
                out_ptr[out_loc] = static_cast<float16_t>(sum);
            }
        }
    });
}

template<typename T>
void quantized_matmul_impl_typed(
    const mx::array &scales, const mx::array &biases,
    const mx::array &a, const mx::array &b,
    mx::array &out, int group_size, int bits, mx::Stream stream) {

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_input_array(a);
    encoder.set_input_array(b);
    encoder.set_output_array(out);

    encoder.dispatch([out_ptr = out.data<T>(), out_shape = out.shape(), out_strides = out.strides(),
        group_size = group_size, bits = bits,
        a = mx::array::unsafe_weak_copy(a), b = mx::array::unsafe_weak_copy(b),
        scales = mx::array::unsafe_weak_copy(scales), biases = mx::array::unsafe_weak_copy(biases)]() {

        // each `group_size` continuous weighted elements are packed into a group and each weight is quantized into `bits` bits
        // thus each `group_size` continuous weighted elements takes `group_size * bits / 32` uint32_t elements in b
        // when decoding the group of weights, the scales and biases are repeated for `group_size` times (shared by all elements in the group)
        int m = a.shape()[0], n = a.shape()[1], k = b.shape()[0];
        
        // row => group => item => pack
        const int group_per_row = n / group_size;   // b[k, :] = [ group_0, group_1, ..., group_(group_per_row-1) ]
        const int packs_per_item = 32 / bits;   // each uint32_t element can store `packs_per_item` packed elements
        const int items_per_group = group_size / packs_per_item;   // each group contains `items_per_group` uint32_t elements

        const T *a_ptr = a.data<T>(),
                *scales_ptr = scales.data<T>(), *biases_ptr = biases.data<T>();
        const uint32_t *b_ptr = b.data<uint32_t>();

        uint32_t pack_mask = (1 << bits) - 1;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                float sum = 0;
                for (int group_idx = 0; group_idx < group_per_row; group_idx++) {
                    int64_t scales_idx = mx::elem_to_loc(j * group_per_row + group_idx, scales.shape(), scales.strides());
                    int64_t biases_idx = mx::elem_to_loc(j * group_per_row + group_idx, biases.shape(), biases.strides());
                    T scale = scales_ptr[scales_idx], bias = biases_ptr[biases_idx];

                    int64_t a_idx = mx::elem_to_loc(i * n + group_idx * group_size, a.shape(), a.strides());
                    int64_t b_idx = mx::elem_to_loc((j * n + group_idx * group_size) / packs_per_item, b.shape(), b.strides());

                    for (int item_idx = 0; item_idx < items_per_group; item_idx++) {
                        uint32_t b_val = b_ptr[b_idx];   // fetch one uint32_t element in current group (item), so we use type uint32_t to store it
                        uint8_t *b_bytes = reinterpret_cast<uint8_t *>(&b_val); // reinterpret the uint32_t element as a byte array (32 = one byte * 4)

                        for (int pack_idx = 0; pack_idx < packs_per_item; pack_idx++) {
                            // extract the pack(4 bits) from the byte array
                            // pack_idx / 2 is the index of the byte array, and (pack_idx % 2) * bits is the shift amount
                            // when pack_idx is even, extract the low 4 bits, otherwise extract the high 4 bits
                            // (pack_7, pack_6, pack_5, pack_4, pack_3, pack_2, pack_1, pack_0) => (b_bytes[3], b_bytes[2], b_bytes[1], b_bytes[0])
                            uint8_t item_val = (b_bytes[pack_idx / 2] >> ((pack_idx % 2) * bits)) & pack_mask;
                            float a_val = static_cast<float>(a_ptr[a_idx]);
                            float b_val_real = static_cast<float>(item_val) * static_cast<float>(scale) + static_cast<float>(bias);
                            sum += a_val * b_val_real;
                            a_idx += 1;
                        }
                        b_idx += 1;
                    }
                }
                int64_t out_idx = mx::elem_to_loc(i * k + j, out_shape, out_strides);
                out_ptr[out_idx] = static_cast<T>(sum);
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

    // quantized_matmul_impl(scales, biases, a, b, out, group_size_, bits_, stream());
    switch (a.dtype()) {
        case mx::float16:
            quantized_matmul_impl_typed<float16_t>(scales, biases, a, b, out, group_size_, bits_, stream());
            break;
        case mx::float32:
            quantized_matmul_impl_typed<float>(scales, biases, a, b, out, group_size_, bits_, stream());
            break;
        case mx::bfloat16:
            quantized_matmul_impl_typed<mx::bfloat16_t>(scales, biases, a, b, out, group_size_, bits_, stream());
            break;
        default:
            throw std::runtime_error("Unsupported dtype for quantized_matmul");
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
    const char* kernel_name;
    if (a.dtype() == mx::float16) {
        kernel_name = "quantized_matmul_w4a16_g64_f16";
    } else if (a.dtype() == mx::bfloat16) {
        kernel_name = "quantized_matmul_w4a16_g64_bf16";
    } else {
        throw std::runtime_error("quantized_matmul: a must be float16 or bfloat16");
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

    if (N % group_size_ != 0) {
        throw std::runtime_error("quantized_matmul: N must be divisible by group_size");
    }

    // Encode matrix parameters
    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);

    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    const int x_size = 32;
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

}  // namespace tiny_llm_ext_ref
