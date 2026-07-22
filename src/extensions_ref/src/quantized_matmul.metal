#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void quantized_matmul_vanilla_w4a16_g128(
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

template <typename T, typename IndexT>
[[kernel]] void quantized_embedding_w4a16_g128(
    device const IndexT* indices [[buffer(0)]],
    device const T* scales [[buffer(1)]],
    device const T* biases [[buffer(2)]],
    device const uint32_t* weights [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &tokens [[buffer(5)]],
    device const int &dim [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    constexpr int bits = 4;
    constexpr int group_size = 128;
    constexpr int packs_per_item = 32 / bits;
    constexpr uint32_t mask = (1 << bits) - 1;
    if (index >= tokens * dim) {
        return;
    }
    const int token = index / dim;
    const int column = index - token * dim;
    const int row = indices[token];
    const int packed_cols = dim / packs_per_item;
    const int groups_per_row = dim / group_size;
    const uint32_t packed =
        weights[row * packed_cols + column / packs_per_item];
    const int shift = (column % packs_per_item) * bits;
    const float quantized = static_cast<float>((packed >> shift) & mask);
    const float scale = static_cast<float>(
        scales[row * groups_per_row + column / group_size]);
    const float bias = static_cast<float>(
        biases[row * groups_per_row + column / group_size]);
    out[index] = static_cast<T>(quantized * scale + bias);
}

// Prefill is a matrix-matrix workload. One SIMD group computes an 8x8
// output tile and advances through the reduction dimension in 8-wide tiles.
// The packed weights are dequantized directly into the matrix fragment, so
// no full-precision weight matrix is ever materialized.
template <typename T>
[[kernel]] void quantized_matmul_simdgroup_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int tile_size = 8;
    constexpr int bits = 4;
    constexpr int group_size = 128;
    constexpr int packs_per_item = 32 / bits;
    constexpr int simdgroups_per_threadgroup = 8;
    constexpr uint32_t mask = (1 << bits) - 1;

    const int row_tiles = (M + tile_size - 1) / tile_size;
    const int column_tiles = (K + tile_size - 1) / tile_size;
    const int tile = group_id * simdgroups_per_threadgroup + simdgroup;
    if (tile >= row_tiles * column_tiles) {
        return;
    }
    const int tile_row = tile / column_tiles;
    const int tile_column = tile - tile_row * column_tiles;
    const int row_base = tile_row * tile_size;
    const int column_base = tile_column * tile_size;

    // In an 8x8 simdgroup matrix, every lane owns two adjacent elements.
    // This coordinate mapping is part of the Metal SIMD matrix layout.
    const int quad = lane / 4;
    const int fragment_row = (quad & 4) + ((lane / 2) % 4);
    const int fragment_column = (quad & 2) * 2 + (lane % 2) * 2;

    // Keep the matrix operands in the model dtype, but accumulate in float.
    // A 16-bit accumulator looks acceptable in a single loose-tolerance
    // kernel test yet compounds visible logit error across all model layers.
    simdgroup_matrix<float, tile_size, tile_size> accumulator;
    #pragma clang loop unroll(full)
    for (int element = 0; element < 2; ++element) {
        accumulator.thread_elements()[element] = 0.0f;
    }

    const int packed_cols = N / packs_per_item;
    const int groups_per_row = N / group_size;
    for (int reduction_base = 0; reduction_base < N; reduction_base += tile_size) {
        simdgroup_matrix<T, tile_size, tile_size> activation_fragment;
        simdgroup_matrix<T, tile_size, tile_size> weight_fragment;
        simdgroup_matrix<float, tile_size, tile_size> next_accumulator;

        #pragma clang loop unroll(full)
        for (int element = 0; element < 2; ++element) {
            const int activation_row = row_base + fragment_row;
            const int reduction_column = reduction_base + fragment_column + element;
            activation_fragment.thread_elements()[element] =
                activation_row < M ? a[activation_row * N + reduction_column] : T(0);

            // The matrix fragment is N x K, while the packed weights are
            // stored as K x N. Transpose while unpacking and dequantizing.
            const int reduction_row = reduction_base + fragment_row;
            const int output_column = column_base + fragment_column + element;
            if (output_column < K) {
                const int quant_group = reduction_row / group_size;
                const int packed_column = reduction_row / packs_per_item;
                const int shift = (reduction_row % packs_per_item) * bits;
                const uint32_t packed = b[output_column * packed_cols + packed_column];
                const float quantized = static_cast<float>((packed >> shift) & mask);
                const float scale = static_cast<float>(
                    scales[output_column * groups_per_row + quant_group]);
                const float bias = static_cast<float>(
                    biases[output_column * groups_per_row + quant_group]);
                weight_fragment.thread_elements()[element] =
                    static_cast<T>(quantized * scale + bias);
            } else {
                weight_fragment.thread_elements()[element] = T(0);
            }
        }
        simdgroup_multiply_accumulate(
            next_accumulator,
            activation_fragment,
            weight_fragment,
            accumulator);
        accumulator = next_accumulator;
    }

    #pragma clang loop unroll(full)
    for (int element = 0; element < 2; ++element) {
        const int output_row = row_base + fragment_row;
        const int output_column = column_base + fragment_column + element;
        if (output_row < M && output_column < K) {
            out[output_row * K + output_column] =
                static_cast<T>(accumulator.thread_elements()[element]);
        }
    }
}

// Decode is a matrix-vector workload: M is usually one and every output row
// reduces over the same activation vector. A full SIMD group cooperates on one
// output instead of assigning the entire reduction to one thread.
template <typename T, int outputs_per_simdgroup>
inline void quantized_matvec_impl(
    device const T* scales,
    device const T* biases,
    device const T* a,
    device const uint32_t* b,
    device T* out,
    device const int &M,
    device const int &N,
    device const int &K,
    uint output_tile,
    uint simdgroup,
    uint lane) {
    constexpr int bits = 4;
    constexpr int group_size = 128;
    constexpr int packs_per_item = 32 / bits;
    constexpr uint32_t mask = (1 << bits) - 1;

    constexpr int simdgroups_per_threadgroup = 8;
    constexpr int outputs_per_threadgroup =
        simdgroups_per_threadgroup * outputs_per_simdgroup;
    const int column_tiles =
        (K + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
    const int row = output_tile / column_tiles;
    const int column_base =
        (output_tile - row * column_tiles) * outputs_per_threadgroup +
        simdgroup * outputs_per_simdgroup;
    if (row >= M) {
        return;
    }

    const int packed_cols = N / packs_per_item;
    const int groups_per_row = N / group_size;
    const int a_base = row * N;
    if (column_base >= K) {
        return;
    }

    float sums[outputs_per_simdgroup] = {0.0f};

    for (int packed_col = lane; packed_col < packed_cols; packed_col += 32) {
        const int group = packed_col / (group_size / packs_per_item);
        const int a_idx = packed_col * packs_per_item;
        float activation_pack[packs_per_item];
        float activation_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (int pack = 0; pack < packs_per_item; ++pack) {
            activation_pack[pack] =
                static_cast<float>(a[a_base + a_idx + pack]);
            if (outputs_per_simdgroup == 8) {
                activation_sum += activation_pack[pack];
            }
        }

        #pragma clang loop unroll(full)
        for (int output = 0; output < outputs_per_simdgroup; ++output) {
            const int column = column_base + output;
            if (column >= K) {
                continue;
            }
            const int params_base = column * groups_per_row;
            const float scale =
                static_cast<float>(scales[params_base + group]);
            const float bias =
                static_cast<float>(biases[params_base + group]);
            const uint32_t packed =
                b[column * packed_cols + packed_col];

            if (outputs_per_simdgroup == 8) {
                float quantized_dot = 0.0f;
                #pragma clang loop unroll(full)
                for (int pack = 0; pack < packs_per_item; ++pack) {
                    quantized_dot += activation_pack[pack] * static_cast<float>(
                        (packed >> (pack * bits)) & mask);
                }
                sums[output] +=
                    scale * quantized_dot + bias * activation_sum;
            } else {
                #pragma clang loop unroll(full)
                for (int pack = 0; pack < packs_per_item; ++pack) {
                    const float weight =
                        static_cast<float>((packed >> (pack * bits)) & mask) *
                            scale +
                        bias;
                    sums[output] += activation_pack[pack] * weight;
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (int output = 0; output < outputs_per_simdgroup; ++output) {
        sums[output] = simd_sum(sums[output]);
    }
    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int output = 0; output < outputs_per_simdgroup; ++output) {
            const int column = column_base + output;
            if (column < K) {
                out[row * K + column] = static_cast<T>(sums[output]);
            }
        }
    }
}

template <typename T>
[[kernel]] void quantized_matvec_x2_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    quantized_matvec_impl<T, 2>(
        scales, biases, a, b, out, M, N, K, output_tile, simdgroup, lane);
}

template <typename T>
[[kernel]] void quantized_matvec_x8_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    quantized_matvec_impl<T, 8>(
        scales, biases, a, b, out, M, N, K, output_tile, simdgroup, lane);
}

instantiate_kernel("quantized_matmul_vanilla_w4a16_g128_f16", quantized_matmul_vanilla_w4a16_g128, half);
instantiate_kernel("quantized_matmul_vanilla_w4a16_g128_bf16", quantized_matmul_vanilla_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matmul_simdgroup_w4a16_g128_f16", quantized_matmul_simdgroup_w4a16_g128, half);
instantiate_kernel("quantized_matmul_simdgroup_w4a16_g128_bf16", quantized_matmul_simdgroup_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_embedding_w4a16_g128_f16_i32", quantized_embedding_w4a16_g128, half, int32_t);
instantiate_kernel("quantized_embedding_w4a16_g128_bf16_i32", quantized_embedding_w4a16_g128, bfloat16_t, int32_t);
instantiate_kernel("quantized_embedding_w4a16_g128_f16_u32", quantized_embedding_w4a16_g128, half, uint32_t);
instantiate_kernel("quantized_embedding_w4a16_g128_bf16_u32", quantized_embedding_w4a16_g128, bfloat16_t, uint32_t);
instantiate_kernel("quantized_matvec_x2_w4a16_g128_f16", quantized_matvec_x2_w4a16_g128, half);
instantiate_kernel("quantized_matvec_x2_w4a16_g128_bf16", quantized_matvec_x2_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matvec_x8_w4a16_g128_f16", quantized_matvec_x8_w4a16_g128, half);
instantiate_kernel("quantized_matvec_x8_w4a16_g128_bf16", quantized_matvec_x8_w4a16_g128, bfloat16_t);
