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

namespace tiny_llm_ext {

mx::array quantized_matmul(const mx::array &scales,         // Input array scales
                           const mx::array &biases,         // Input array biases
                           const int group_size,            // Group size
                           const int bits,                  // Number of bits
                           const mx::array &a,              // Input array a (not quantized)
                           const mx::array &b,              // Input array b (quantized)
                           const bool transpose_b,          // Whether to transpose b
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

    return mx::array(
        /* const mx::Shape& shape = */ out_shape,
        /* mx::Dtype dtype = */ a.dtype(),
        /* std::shared_ptr<mx::Primitive> primitive = */
        std::make_shared<QuantizedMatmul>(to_stream(s), group_size, bits),
        /* const std::vector<mx::array>& inputs = */ {scales, biases, a, b});    
}


void quantized_matmul_impl(const mx::array &scales, const mx::array &biases, const mx::array &a, const mx::array &b,
                           mx::array &out, const int group_size, const int bits, mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // Get the CPU command encoder and register input and output arrays
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

    // Launch the CPU kernel
    encoder.dispatch([
        out_ptr = out.data<float16_t>(),
        out_shape = out.shape(),
        out_strides = out.strides(),
        scales = mx::array::unsafe_weak_copy(scales),
        biases = mx::array::unsafe_weak_copy(biases),
        a = mx::array::unsafe_weak_copy(a),
        b = mx::array::unsafe_weak_copy(b),
        group_size, bits
    ](){
        int M = a.shape()[0];
        int N = a.shape()[1];
        int K = b.shape()[0];
        int group_per_row = N / group_size;
        int pack_factor = 32 / bits;
        uint32_t item_mask = (1 << bits) - 1;

        const float16_t* scales_ptr = scales.data<float16_t>();
        const float16_t* biases_ptr = biases.data<float16_t>();
        const float16_t* a_ptr = a.data<float16_t>();
        const uint32_t* b_ptr = b.data<uint32_t>();

        // Do the element-wise operation for each output
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float sum = 0; 
                for (int g = 0; g < group_per_row; g++) {
                    auto scales_loc = mx::elem_to_loc(k * group_per_row + g, scales);
                    auto bias_loc = mx::elem_to_loc(k * group_per_row + g, biases);
                    auto a_elem = i * N + g * group_size;
                    // b stores 8x 4-bit values per uint32; convert element offset to packed-word offset.
                    auto b_elem = (k * N + g * group_size) / pack_factor;
                    for (int word = 0; word < group_size/pack_factor; word++) {
                        uint32_t packed = b_ptr[mx::elem_to_loc(b_elem + word, b)];
                        for (int pack_idx = 0; pack_idx < pack_factor; pack_idx++) {
                            auto shift = (pack_idx * bits);
                            auto quantized_val = (packed >> shift) & item_mask;
                            float dequantized_val = static_cast<float16_t>(quantized_val) * scales_ptr[scales_loc] + biases_ptr[bias_loc];
                            float a_val = a_ptr[mx::elem_to_loc(a_elem + word * pack_factor + pack_idx, a)];
                            sum += a_val * dequantized_val;
                        }
                    }
                }
                
                auto out_idx = mx::elem_to_loc(i * K + k, out_shape, out_strides);
                out_ptr[out_idx] = static_cast<float16_t>(sum);
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

    quantized_matmul_impl(scales, biases, a, b, out, group_size_, bits_, stream());
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

    auto lib = d.get_library("tiny_llm_ext");
    const char* kname;
    if (a.dtype() == mx::float16) {
        kname = "quantized_matmul_w4a16_g64_f16"; 
    } else if (a.dtype() == mx::bfloat16) {
        kname = "quantized_matmul_w4a16_g64_bf16";
    } else {
        throw std::runtime_error("quantized_matmul: a must be float16 or bfloat16");
    }
    auto kernel = d.get_kernel(kname, lib);

    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(scales, 0);
    compute_encoder.set_input_array(biases, 1);
    compute_encoder.set_input_array(a, 2);
    compute_encoder.set_input_array(b, 3);
    compute_encoder.set_output_array(out, 4);

    int M = a.shape()[0];
    int N = a.shape()[1];
    int K = b.shape()[0];
    if (N % group_size_ != 0) {
        throw std::runtime_error("quantized_matmul: N must be divisible by group_size");
    }

    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);

    size_t tpg_size = kernel->maxTotalThreadsPerThreadgroup();
    int x_size = kernel->threadExecutionWidth();
    int y_size = tpg_size / x_size;

    MTL::Size num_threadgroups = MTL::Size((M + x_size - 1) / x_size, (K + y_size - 1) / y_size, 1);
    MTL::Size num_threads_per_group = MTL::Size(x_size, y_size, 1);
    compute_encoder.dispatch_threadgroups(num_threadgroups, num_threads_per_group); 
}

} // namespace tiny_llm_ext
