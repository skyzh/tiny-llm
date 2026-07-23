#include <algorithm>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
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
                           const bool use_simdgroup, const bool use_split_k,
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
        std::make_shared<QuantizedMatmul>(to_stream(s), use_simdgroup, use_split_k),
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

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("quantized_matmul: the course extension is GPU-only");
}

void QuantizedEmbedding::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("quantized_embedding: the course extension is GPU-only");
}

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &scales = inputs[0];
    auto &biases = inputs[1];
    auto &a = inputs[2];
    auto &b = inputs[3];
    auto &out = outputs[0];

    if (!a.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: a must be contiguous");
    }
    if (!b.flags().row_contiguous) {
        throw std::runtime_error("quantized_matmul: b must be contiguous");
    }

    const int M = a.shape()[0];
    const int N = a.shape()[1];
    const int K = b.shape()[0];
    if (N % 128 != 0) {
        throw std::runtime_error("quantized_matmul: N must be divisible by group_size");
    }

    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    auto library = d.get_library("tiny_llm_ext_ref");
    out.set_data(mx::allocator::malloc(out.nbytes()));

    const bool use_matvec = use_simdgroup_ && M <= 8;
    const bool use_streaming_matvec = use_matvec && N % 512 == 0 && K % 8 == 0;
    int split_k = 1;
    if (use_split_k_ && use_simdgroup_ && !use_matvec) {
        constexpr int block_size = 32;
        constexpr int target_threadgroups = 320;
        constexpr int max_split_k = 16;
        const int row_blocks = (M + block_size - 1) / block_size;
        const int column_blocks = (K + block_size - 1) / block_size;
        const int threadgroups = row_blocks * column_blocks;
        split_k = std::min({max_split_k, std::max(1, target_threadgroups / std::max(threadgroups, 1)), N / 128});
        while (split_k > 1 && N % (split_k * 128) != 0) {
            split_k--;
        }
    }
    const bool use_split_k = split_k > 1;

    const char *kernel_name;
    if (use_matvec) {
        if (use_streaming_matvec) {
            kernel_name = out.dtype() == mx::float16 ? "quantized_matvec_x4_streaming_w4a16_g128_f16"
                                                     : "quantized_matvec_x4_streaming_w4a16_g128_bf16";
        } else {
            kernel_name = out.dtype() == mx::float16 ? "quantized_matvec_x4_fast_w4a16_g128_f16"
                                                     : "quantized_matvec_x4_fast_w4a16_g128_bf16";
        }
    } else if (use_split_k) {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matmul_simdgroup_splitk_w4a16_g128_f16"
                                                 : "quantized_matmul_simdgroup_splitk_w4a16_g128_bf16";
    } else if (use_simdgroup_) {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matmul_simdgroup_w4a16_g128_f16"
                                                 : "quantized_matmul_simdgroup_w4a16_g128_bf16";
    } else {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matmul_vanilla_w4a16_g128_f16"
                                                 : "quantized_matmul_vanilla_w4a16_g128_bf16";
    }
    auto kernel = d.get_kernel(kernel_name, library);
    auto &compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(scales, 0);
    compute_encoder.set_input_array(biases, 1);
    compute_encoder.set_input_array(a, 2);
    compute_encoder.set_input_array(b, 3);

    if (use_split_k) {
        auto partial_shape = out.shape();
        partial_shape.insert(partial_shape.begin(), split_k);
        mx::array partials(partial_shape, out.dtype(), nullptr, {});
        partials.set_data(mx::allocator::malloc(partials.nbytes()));
        compute_encoder.add_temporary(partials);
        compute_encoder.set_output_array(partials, 4);
        compute_encoder.set_bytes(M, 5);
        compute_encoder.set_bytes(N, 6);
        compute_encoder.set_bytes(K, 7);
        const int partition_size = N / split_k;
        const int partition_stride = M * K;
        compute_encoder.set_bytes(partition_size, 8);
        compute_encoder.set_bytes(partition_stride, 9);

        constexpr int block_size = 32;
        const int row_blocks = (M + block_size - 1) / block_size;
        const int column_blocks = (K + block_size - 1) / block_size;
        compute_encoder.dispatch_threadgroups(MTL::Size(column_blocks, row_blocks, split_k), MTL::Size(128, 1, 1));

        const char *reduce_name =
            out.dtype() == mx::float16 ? "quantized_matmul_splitk_reduce_f16" : "quantized_matmul_splitk_reduce_bf16";
        auto reduce_kernel = d.get_kernel(reduce_name, library);
        compute_encoder.set_compute_pipeline_state(reduce_kernel);
        compute_encoder.set_input_array(partials, 0);
        compute_encoder.set_output_array(out, 1);
        const int elements = M * K;
        compute_encoder.set_bytes(elements, 2);
        compute_encoder.set_bytes(split_k, 3);
        const int threads = std::min<int>(reduce_kernel->maxTotalThreadsPerThreadgroup(), 256);
        compute_encoder.dispatch_threads(MTL::Size(elements, 1, 1), MTL::Size(threads, 1, 1));
        return;
    }

    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_bytes(M, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(K, 7);

    if (use_matvec) {
        constexpr int outputs_per_simdgroup = 4;
        constexpr int simdgroups_per_threadgroup = 2;
        const int outputs_per_threadgroup = simdgroups_per_threadgroup * outputs_per_simdgroup;
        const int column_tiles = (K + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
        const auto grid = use_streaming_matvec ? MTL::Size(column_tiles, M, 1)
                                               : MTL::Size(M * column_tiles, 1, 1);
        compute_encoder.dispatch_threadgroups(grid, MTL::Size(simdgroups_per_threadgroup * 32, 1, 1));
        return;
    }

    if (use_simdgroup_) {
        constexpr int block_size = 32;
        const int row_blocks = (M + block_size - 1) / block_size;
        const int column_blocks = (K + block_size - 1) / block_size;
        compute_encoder.dispatch_threadgroups(MTL::Size(column_blocks, row_blocks, 1), MTL::Size(128, 1, 1));
        return;
    }

    const size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    const int x_size = M <= 16 ? 16 : 32;
    const int y_size = tgp_size / x_size;
    if (tgp_size < x_size * y_size) {
        throw std::runtime_error("quantized_matmul: tgp_size must be larger than x*y");
    }
    compute_encoder.dispatch_threadgroups(MTL::Size((M + x_size - 1) / x_size, (K + y_size - 1) / y_size, 1),
                                          MTL::Size(x_size, y_size, 1));
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
    auto &encoder = mx::metal::get_command_encoder(stream());
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
