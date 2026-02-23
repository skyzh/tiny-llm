#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void quantized_matmul_w4a16_g64(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {

    const int group_size = 64;
    const int bits = 4;
    const int pack_factor = 32 / bits; // number of quantized values per uint
    const int group_per_row = N / group_size;
    const int item_mask = (1 << bits) - 1;

    // Each thread processes an element in the output matrix
    const int i = group_id.x * threads_per_threadgroup.x + thread_id.x;
    const int k = group_id.y * threads_per_threadgroup.y + thread_id.y;

    float sum = 0;
    if (i < M && k < K) {
        // dequantize b = quantized_b * scale + bias
        for (int g = 0; g < group_per_row; g++) {
            auto scales_loc = k * group_per_row + g;
            auto bias_loc = k * group_per_row + g;
            auto a_base_loc = i * N + g * group_size;
            // b stores 8x 4-bit values per uint32; convert element offset to packed-word offset.
            auto b_base_loc = (k * N + g * group_size) / pack_factor;
            for (int word = 0; word < group_size/pack_factor; word++) {
                uint32_t packed = b[b_base_loc + word];
                for (int pack_idx = 0; pack_idx < pack_factor; pack_idx++) {
                    auto shift = (pack_idx * bits);
                    auto quantized_val = (packed >> shift) & item_mask;
                    float dequantized_val = static_cast<float16_t>(quantized_val) * scales[scales_loc] + biases[bias_loc];
                    float a_val = a[a_base_loc + word * pack_factor + pack_idx];
                    sum += a_val * dequantized_val;
                }
            }
        }
            
        out[i * K + k] = static_cast<T>(sum);
    }
}

instantiate_kernel("quantized_matmul_w4a16_g64_f16", quantized_matmul_w4a16_g64, float16_t);