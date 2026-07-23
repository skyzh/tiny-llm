#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"

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

// Prefill uses a 32x32 threadgroup tile. Four SIMD groups cooperatively load
// one activation tile and dequantize one weight tile into threadgroup memory;
// each group then computes a 16x16 quadrant with four 8x8 matrix fragments.
// This reuses each device-memory load across 32 output values without ever
// materializing a full dequantized weight matrix.
template <typename T, typename OutT>
inline void quantized_matmul_block_w4a16_g128(
    device const T* scales,
    device const T* biases,
    device const T* a,
    device const uint32_t* b,
    device OutT* out,
    const int M,
    const int N,
    const int K,
    const int reduction_start,
    const int reduction_end,
    const int output_offset,
    threadgroup T* activation_tile,
    threadgroup T* weight_tile,
    threadgroup T* quantization_parameters,
    uint3 group_id,
    uint thread_id,
    uint simdgroup,
    uint lane) {
    constexpr int output_block_size = 32;
    constexpr int reduction_block_size = 32;
    constexpr int padded_reduction_size = 40;
    constexpr int group_size = 128;
    constexpr int packs_per_item = 8;
    constexpr uint32_t mask = 0xf;
    const int row_base = group_id.y * output_block_size;
    const int column_base = group_id.x * output_block_size;
    const int packed_cols = N / packs_per_item;
    const int groups_per_row = N / group_size;

    using block_mma = mlx::steel::BlockMMA<
        T, OutT, output_block_size, output_block_size,
        reduction_block_size, 2, 2, false, true,
        padded_reduction_size, padded_reduction_size>;
    using activation_loader = mlx::steel::BlockLoader<
        T, output_block_size, reduction_block_size,
        padded_reduction_size, 1, 128>;
    block_mma mma(simdgroup, lane);
    activation_loader load_activation(
        a + row_base * N + reduction_start,
        N,
        activation_tile,
        simdgroup,
        lane);

    const int weight_output = thread_id / 4;
    const int weight_pack = thread_id % 4;
    const int output_column = column_base + weight_output;
    const bool valid_output = output_column < K;
    device const uint32_t* weight_source = valid_output
        ? b + output_column * packed_cols + reduction_start / packs_per_item +
            weight_pack
        : b;
    threadgroup T* weight_destination =
        weight_tile + weight_output * padded_reduction_size +
        weight_pack * packs_per_item;
    int quantization_group_step = 0;

    // One thread loads the scale and bias for each output column. The four
    // threads that unpack that column's 32 weights then reuse the values from
    // threadgroup memory for all four reduction tiles in the quantization
    // group.
    if (thread_id < output_block_size) {
        const int parameter_output = column_base + thread_id;
        const bool valid_parameter = parameter_output < K;
        const int parameter_index =
            parameter_output * groups_per_row + reduction_start / group_size;
        quantization_parameters[thread_id] =
            valid_parameter ? scales[parameter_index] : T(0);
        quantization_parameters[output_block_size + thread_id] =
            valid_parameter ? biases[parameter_index] : T(0);
    }

    for (int reduction_base = reduction_start;
         reduction_base < reduction_end;
         reduction_base += reduction_block_size) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_base + output_block_size <= M) {
            load_activation.load_unsafe();
        } else {
            load_activation.load_safe(short2(
                reduction_block_size,
                max(0, M - row_base)));
        }

        const uint32_t packed = valid_output ? *weight_source : 0;
        const float scale =
            static_cast<float>(quantization_parameters[weight_output]);
        const float bias = static_cast<float>(
            quantization_parameters[output_block_size + weight_output]);
        #pragma clang loop unroll(full)
        for (int value = 0; value < packs_per_item; ++value) {
            const float quantized =
                static_cast<float>((packed >> (value * 4)) & mask);
            weight_destination[value] =
                static_cast<T>(quantized * scale + bias);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.mma(activation_tile, weight_tile);

        load_activation.next();
        weight_source += reduction_block_size / packs_per_item;
        quantization_group_step += reduction_block_size;
        if (quantization_group_step == group_size) {
            quantization_group_step = 0;
            const int next_reduction_base =
                reduction_base + reduction_block_size;
            if (next_reduction_base < reduction_end &&
                thread_id < output_block_size) {
                const int parameter_output = column_base + thread_id;
                const bool valid_parameter = parameter_output < K;
                const int parameter_index = parameter_output * groups_per_row +
                    next_reduction_base / group_size;
                quantization_parameters[thread_id] =
                    valid_parameter ? scales[parameter_index] : T(0);
                quantization_parameters[output_block_size + thread_id] =
                    valid_parameter ? biases[parameter_index] : T(0);
            }
        }
    }

    const short valid_rows = min(output_block_size, M - row_base);
    const short valid_columns = min(output_block_size, K - column_base);
    mma.store_result_safe(
        out + output_offset + row_base * K + column_base,
        K,
        short2(valid_columns, valid_rows));
}

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
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup T activation_tile[32 * 40];
    threadgroup T weight_tile[32 * 40];
    threadgroup T quantization_parameters[2 * 32];
    quantized_matmul_block_w4a16_g128(
        scales, biases, a, b, out, M, N, K, 0, N, 0,
        activation_tile, weight_tile, quantization_parameters,
        group_id, thread_id, simdgroup, lane);
}

template <typename T>
[[kernel]] void quantized_matmul_simdgroup_splitk_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* partials [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    device const int &partition_size [[buffer(8)]],
    device const int &partition_stride [[buffer(9)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup T activation_tile[32 * 40];
    threadgroup T weight_tile[32 * 40];
    threadgroup T quantization_parameters[2 * 32];
    const int reduction_start = group_id.z * partition_size;
    quantized_matmul_block_w4a16_g128(
        scales, biases, a, b, partials, M, N, K,
        reduction_start, reduction_start + partition_size,
        group_id.z * partition_stride, activation_tile, weight_tile,
        quantization_parameters, group_id, thread_id, simdgroup, lane);
}

template <typename T>
[[kernel]] void quantized_matmul_splitk_reduce(
    device const T* partials [[buffer(0)]],
    device T* out [[buffer(1)]],
    device const int &elements [[buffer(2)]],
    device const int &split_k [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= static_cast<uint>(elements)) {
        return;
    }
    float sum = 0.0f;
    for (int partition = 0; partition < split_k; ++partition) {
        sum += static_cast<float>(partials[partition * elements + index]);
    }
    out[index] = static_cast<T>(sum);
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

// Qwen's ordinary decode projections use the same four-output, two-packed-word
// schedule as the measured MLX qmv shape. Loading 16 adjacent activation
// values once and reusing them across four weight rows cuts SIMD-group and
// activation-load work without changing the course W4A16 arithmetic.
template <typename T>
[[kernel]] void quantized_matvec_x4_fast_w4a16_g128(
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
    constexpr int group_size = 128;
    constexpr int packs_per_item = 8;
    constexpr int packs_per_lane = 2;
    constexpr int values_per_lane = packs_per_item * packs_per_lane;
    constexpr int outputs_per_simdgroup = 4;
    constexpr int simdgroups_per_threadgroup = 2;
    constexpr int outputs_per_threadgroup =
        outputs_per_simdgroup * simdgroups_per_threadgroup;

    const int column_tiles =
        (K + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
    const int row = output_tile / column_tiles;
    const int column_base =
        (output_tile - row * column_tiles) * outputs_per_threadgroup +
        simdgroup * outputs_per_simdgroup;
    if (row >= M || column_base >= K) return;

    const int packed_cols = N / packs_per_item;
    const int groups_per_row = N / group_size;
    const int activation_base = row * N;
    float sums[outputs_per_simdgroup] = {0.0f};

    for (int packed_col = lane * packs_per_lane;
         packed_col < packed_cols;
         packed_col += 32 * packs_per_lane) {
        const int group = packed_col / (group_size / packs_per_item);
        float scaled_activations[values_per_lane];
        float activation_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (int pack = 0; pack < packs_per_lane; ++pack) {
            const int activation_offset =
                activation_base + (packed_col + pack) * packs_per_item;
            #pragma clang loop unroll(full)
            for (int value = 0; value < packs_per_item; ++value) {
                const int local = pack * packs_per_item + value;
                const float activation =
                    static_cast<float>(a[activation_offset + value]);
                activation_sum += activation;
                // Four adjacent W4 values occupy one uint16. Scale the
                // activations once so the hot loop can use masks directly
                // instead of shifting every weight for every output row.
                scaled_activations[local] = activation /
                    static_cast<float>(1 << ((value & 3) * 4));
            }
        }

        #pragma clang loop unroll(full)
        for (int output = 0; output < outputs_per_simdgroup; ++output) {
            const int column = column_base + output;
            if (column >= K) continue;
            const int parameter_index = column * groups_per_row + group;
            const float scale = static_cast<float>(scales[parameter_index]);
            const float bias = static_cast<float>(biases[parameter_index]);
            const device uint16_t* packed =
                reinterpret_cast<const device uint16_t*>(
                    b + column * packed_cols + packed_col);
            float quantized_dot = 0.0f;
            #pragma clang loop unroll(full)
            for (int group = 0; group < values_per_lane / 4; ++group) {
                const uint16_t weights = packed[group];
                const int local = group * 4;
                quantized_dot +=
                    scaled_activations[local] * (weights & 0x000f) +
                    scaled_activations[local + 1] * (weights & 0x00f0) +
                    scaled_activations[local + 2] * (weights & 0x0f00) +
                    scaled_activations[local + 3] * (weights & 0xf000);
            }
            sums[output] += scale * quantized_dot + bias * activation_sum;
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

instantiate_kernel("quantized_matmul_vanilla_w4a16_g128_f16", quantized_matmul_vanilla_w4a16_g128, half);
instantiate_kernel("quantized_matmul_vanilla_w4a16_g128_bf16", quantized_matmul_vanilla_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matmul_simdgroup_w4a16_g128_f16", quantized_matmul_simdgroup_w4a16_g128, half);
instantiate_kernel("quantized_matmul_simdgroup_w4a16_g128_bf16", quantized_matmul_simdgroup_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matmul_simdgroup_splitk_w4a16_g128_f16", quantized_matmul_simdgroup_splitk_w4a16_g128, half);
instantiate_kernel("quantized_matmul_simdgroup_splitk_w4a16_g128_bf16", quantized_matmul_simdgroup_splitk_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matmul_splitk_reduce_f16", quantized_matmul_splitk_reduce, half);
instantiate_kernel("quantized_matmul_splitk_reduce_bf16", quantized_matmul_splitk_reduce, bfloat16_t);
instantiate_kernel("quantized_embedding_w4a16_g128_f16_i32", quantized_embedding_w4a16_g128, half, int32_t);
instantiate_kernel("quantized_embedding_w4a16_g128_bf16_i32", quantized_embedding_w4a16_g128, bfloat16_t, int32_t);
instantiate_kernel("quantized_embedding_w4a16_g128_f16_u32", quantized_embedding_w4a16_g128, half, uint32_t);
instantiate_kernel("quantized_embedding_w4a16_g128_bf16_u32", quantized_embedding_w4a16_g128, bfloat16_t, uint32_t);
instantiate_kernel("quantized_matvec_x2_w4a16_g128_f16", quantized_matvec_x2_w4a16_g128, half);
instantiate_kernel("quantized_matvec_x2_w4a16_g128_bf16", quantized_matvec_x2_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matvec_x8_w4a16_g128_f16", quantized_matvec_x8_w4a16_g128, half);
instantiate_kernel("quantized_matvec_x8_w4a16_g128_bf16", quantized_matvec_x8_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matvec_x4_fast_w4a16_g128_f16", quantized_matvec_x4_fast_w4a16_g128, half);
instantiate_kernel("quantized_matvec_x4_fast_w4a16_g128_bf16", quantized_matvec_x4_fast_w4a16_g128, bfloat16_t);
