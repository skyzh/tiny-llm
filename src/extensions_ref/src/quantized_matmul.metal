#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void quantized_matmul_w4a16_g128(
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
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    [[maybe_unused]] threadgroup char * shmem [[threadgroup(0)]]) {
    const int bits = 4;
    const int group_size = 128;
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
                sum += (static_cast<float>((b_val_packed >> 0) & mask) * scale + bias) * static_cast<float>(a[a_loc]);
                sum += (static_cast<float>((b_val_packed >> 4) & mask) * scale + bias) * static_cast<float>(a[a_loc + 1]);
                sum += (static_cast<float>((b_val_packed >> 8) & mask) * scale + bias) * static_cast<float>(a[a_loc + 2]);
                sum += (static_cast<float>((b_val_packed >> 12) & mask) * scale + bias) * static_cast<float>(a[a_loc + 3]);
                sum += (static_cast<float>((b_val_packed >> 16) & mask) * scale + bias) * static_cast<float>(a[a_loc + 4]);
                sum += (static_cast<float>((b_val_packed >> 20) & mask) * scale + bias) * static_cast<float>(a[a_loc + 5]);
                sum += (static_cast<float>((b_val_packed >> 24) & mask) * scale + bias) * static_cast<float>(a[a_loc + 6]);
                sum += (static_cast<float>((b_val_packed >> 28) & mask) * scale + bias) * static_cast<float>(a[a_loc + 7]);
                a_loc += packs_per_item;
                b_loc += 1;
            }
            scales_biases_loc += 1;
        }
        out[i * K + k] = static_cast<T>(sum);
    }
}

// Decode is a matrix-vector workload: M is usually one and every output row
// reduces over the same activation vector. A full SIMD group cooperates on one
// output instead of assigning the entire reduction to one thread.
template <typename T>
[[kernel]] void quantized_matvec_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    threadgroup T* activations [[threadgroup(0)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int bits = 4;
    constexpr int group_size = 128;
    constexpr int packs_per_item = 32 / bits;
    constexpr uint32_t mask = (1 << bits) - 1;

    constexpr int simdgroups_per_threadgroup = 8;
    const int column_tiles = (K + simdgroups_per_threadgroup - 1) / simdgroups_per_threadgroup;
    const int row = output_tile / column_tiles;
    const int column = (output_tile - row * column_tiles) * simdgroups_per_threadgroup + simdgroup;
    if (row >= M) {
        return;
    }

    const int packed_cols = N / packs_per_item;
    const int groups_per_row = N / group_size;
    const int a_base = row * N;
    for (int d = thread_index; d < N; d += simdgroups_per_threadgroup * 32) {
        activations[d] = a[a_base + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (column >= K) {
        return;
    }

    float sum = 0.0f;
    const int params_base = column * groups_per_row;

    for (int packed_col = lane; packed_col < packed_cols; packed_col += 32) {
        const int group = packed_col / (group_size / packs_per_item);
        const int a_idx = packed_col * packs_per_item;
        const float scale = static_cast<float>(scales[params_base + group]);
        const float bias = static_cast<float>(biases[params_base + group]);
        const uint32_t packed = b[column * packed_cols + packed_col];

        #pragma clang loop unroll(full)
        for (int pack = 0; pack < packs_per_item; ++pack) {
            const float weight = static_cast<float>((packed >> (pack * bits)) & mask) * scale + bias;
            sum += static_cast<float>(activations[a_idx + pack]) * weight;
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        out[row * K + column] = static_cast<T>(sum);
    }
}

instantiate_kernel("quantized_matmul_w4a16_g128_f16", quantized_matmul_w4a16_g128, half);
instantiate_kernel("quantized_matmul_w4a16_g128_bf16", quantized_matmul_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matvec_w4a16_g128_f16", quantized_matvec_w4a16_g128, half);
instantiate_kernel("quantized_matvec_w4a16_g128_bf16", quantized_matvec_w4a16_g128, bfloat16_t);
