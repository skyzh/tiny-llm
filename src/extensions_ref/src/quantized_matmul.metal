#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

METAL_FUNC float round_to_bfloat16(float value) {
    uint bits = as_type<uint>(value);
    uint rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
    bits = (bits + rounding_bias) & 0xffff0000u;
    return as_type<float>(bits);
}

METAL_FUNC float accumulate_quantized_product(
    float sum, uint32_t quantized, float scale, float bias, float a_value) {
    float b_value = round_to_bfloat16(static_cast<float>(quantized) * scale + bias);
    float product = round_to_bfloat16(a_value * b_value);
    return round_to_bfloat16(sum + product);
}

template <typename T>
[[kernel]] void quantized_matmul_w4a16(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    device const int &group_size [[buffer(8)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    [[maybe_unused]] threadgroup char * shmem [[threadgroup(0)]]) {
    const int bits = 4;
    const int packs_per_item = 32 / bits;
    const int groups_per_row = N / group_size;
    // Each thread processes an element in the output matrix
    const int i = group_id.x * threads_per_threadgroup.x + thread_id.x;
    const int k = group_id.y * threads_per_threadgroup.y + thread_id.y;
    float sum = 0;
    int scales_biases_loc = k * groups_per_row;
    const int mask = (1 << bits) - 1;
    // A: M * N, B: K * N where N gets quantized
    if (i < M && k < K) {
        int b_loc = k * N / packs_per_item;
        int a_loc = i * N;
        for (int group_idx = 0; group_idx < groups_per_row; group_idx++) {
            const float scale = scales[scales_biases_loc];
            const float bias = biases[scales_biases_loc];
            for (int item_idx = 0; item_idx < group_size; item_idx += packs_per_item) {
                uint32_t b_val_packed = b[b_loc];
                sum = accumulate_quantized_product(sum, (b_val_packed >> 0) & mask, scale, bias, a[a_loc]);
                sum = accumulate_quantized_product(sum, (b_val_packed >> 4) & mask, scale, bias, a[a_loc + 1]);
                sum = accumulate_quantized_product(sum, (b_val_packed >> 8) & mask, scale, bias, a[a_loc + 2]);
                sum = accumulate_quantized_product(sum, (b_val_packed >> 12) & mask, scale, bias, a[a_loc + 3]);
                sum = accumulate_quantized_product(sum, (b_val_packed >> 16) & mask, scale, bias, a[a_loc + 4]);
                sum = accumulate_quantized_product(sum, (b_val_packed >> 20) & mask, scale, bias, a[a_loc + 5]);
                sum = accumulate_quantized_product(sum, (b_val_packed >> 24) & mask, scale, bias, a[a_loc + 6]);
                sum = accumulate_quantized_product(sum, (b_val_packed >> 28) & mask, scale, bias, a[a_loc + 7]);
                a_loc += packs_per_item;
                b_loc += 1;
            }
            scales_biases_loc += 1;
        }
        out[i * K + k] = static_cast<T>(sum);
    }
}

instantiate_kernel("quantized_matmul_w4a16_bf16", quantized_matmul_w4a16, bfloat16_t);
