#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "mlx/backend/metal/kernels/utils.h"
#include "cooperative_matrix.h"

using namespace metal;

template <typename T>
[[kernel]] void week2_rms_norm(
    device const T* x [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& dim [[buffer(4)]],
    constant const float& eps [[buffer(5)]],
    threadgroup float* partial_sums [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= rows) return;
    constexpr int threads_per_threadgroup = 256;
    constexpr int simdgroups_per_threadgroup = threads_per_threadgroup / 32;
    float sum = 0.0f;
    for (int col = thread_index; col < dim; col += threads_per_threadgroup) {
        const float value = static_cast<float>(x[row * dim + col]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        partial_sums[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float threadgroup_sum =
            lane < simdgroups_per_threadgroup ? partial_sums[lane] : 0.0f;
        threadgroup_sum = simd_sum(threadgroup_sum);
        if (lane == 0) {
            partial_sums[0] = threadgroup_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv =
        rsqrt(partial_sums[0] / static_cast<float>(dim) + eps);
    for (int col = thread_index; col < dim; col += threads_per_threadgroup) {
        out[row * dim + col] = static_cast<T>(
            static_cast<float>(x[row * dim + col]) * inv *
            static_cast<float>(weight[col]));
    }
}

template <typename T>
[[kernel]] void week2_rope(
    device const T* x [[buffer(0)]],
    device const int32_t* offsets [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& batch [[buffer(3)]],
    constant const int& length [[buffer(4)]],
    constant const int& heads [[buffer(5)]],
    constant const int& head_dim [[buffer(6)]],
    constant const int& dims [[buffer(7)]],
    constant const float& base [[buffer(8)]],
    constant const int& traditional [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    constexpr int heads_per_thread = 4;
    const int half_dim = dims / 2;
    const int tail_dims = head_dim - dims;
    const int items_per_head_block = half_dim + tail_dims;
    const int head_blocks = (heads + heads_per_thread - 1) / heads_per_thread;
    const int total = batch * length * head_blocks * items_per_head_block;
    if (index >= total) return;
    const int item = index % items_per_head_block;
    const int head_block = (index / items_per_head_block) % head_blocks;
    const int l = (index / (items_per_head_block * head_blocks)) % length;
    const int b = index / (items_per_head_block * head_blocks * length);
    const int first_head = head_block * heads_per_thread;
    const int last_head = min(first_head + heads_per_thread, heads);
    const int row_base = (b * length + l) * heads * head_dim;

    if (item >= half_dim) {
        const int d = dims + item - half_dim;
        for (int h = first_head; h < last_head; ++h) {
            const int element = row_base + h * head_dim + d;
            out[element] = x[element];
        }
        return;
    }
    const int pair = item;
    const float frequency_power = -static_cast<float>(pair) / half_dim;
    const float angle = sizeof(T) == sizeof(float)
        ? static_cast<float>(offsets[b] + l) * pow(base, frequency_power)
        : static_cast<float>(offsets[b] + l) *
            fast::exp2(frequency_power * log2(base));
    const float c =
        sizeof(T) == sizeof(float) ? cos(angle) : fast::cos(angle);
    const float s =
        sizeof(T) == sizeof(float) ? sin(angle) : fast::sin(angle);
    for (int h = first_head; h < last_head; ++h) {
        const int head_base = row_base + h * head_dim;
        const int real_idx = traditional ? head_base + pair * 2 : head_base + pair;
        const int imag_idx = traditional ? real_idx + 1 : real_idx + half_dim;
        const float real = static_cast<float>(x[real_idx]);
        const float imag = static_cast<float>(x[imag_idx]);
        out[real_idx] = static_cast<T>(real * c - imag * s);
        out[imag_idx] = static_cast<T>(imag * c + real * s);
    }
}

template <typename T>
[[kernel]] void week2_swiglu(
    device const T* gate [[buffer(0)]],
    device const T* up [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& size [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= size) return;
    const float g = static_cast<float>(gate[index]);
    out[index] = static_cast<T>((g / (1.0f + exp(-g))) * static_cast<float>(up[index]));
}

// Gate and up share the same activation tile. Two FP32 matrix accumulators
// consume independent packed-W4 tiles, then SwiGLU is applied before the
// single BF16 store. Edge guards cover row and intermediate-size tails.
template <typename T>
[[kernel]] void week2_quantized_gate_up_swiglu(
    device const T* x [[buffer(0)]],
    device const T* gate_scales [[buffer(1)]],
    device const T* gate_biases [[buffer(2)]],
    device const uint32_t* gate_weight [[buffer(3)]],
    device const T* up_scales [[buffer(4)]],
    device const T* up_biases [[buffer(5)]],
    device const uint32_t* up_weight [[buffer(6)]],
    device T* out [[buffer(7)]],
    constant const int& rows [[buffer(8)]],
    constant const int& input_dim [[buffer(9)]],
    constant const int& output_dim [[buffer(10)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int block = 32;
    constexpr int reduction = 32;
    constexpr int padded_reduction = 40;
    constexpr int group_size = 128;
    constexpr int values_per_pack = 8;
    constexpr uint32_t mask = 0xf;
    const int row_base = group_id.y * block;
    const int column_base = group_id.x * block;
    const int packed_cols = input_dim / values_per_pack;
    const int groups_per_output = input_dim / group_size;

    threadgroup T activation_tile[block * padded_reduction];
    threadgroup T gate_tile[block * padded_reduction];
    threadgroup T up_tile[block * padded_reduction];
    threadgroup T parameters[4 * block];
    using mma_type = tiny_llm::CooperativeBlockMMA<T, T, padded_reduction>;
    using activation_loader = tiny_llm::CooperativeTileLoader<
        T, block, reduction, padded_reduction, 128, false, true>;
    mma_type gate_mma(simdgroup, lane);
    mma_type up_mma(simdgroup, lane);

    const int weight_output = thread_index / 4;
    const int weight_pack = thread_index % 4;
    const int output_column = column_base + weight_output;
    const bool valid_output = output_column < output_dim;
    device const uint32_t* gate_source = valid_output
        ? gate_weight + output_column * packed_cols + weight_pack
        : gate_weight;
    device const uint32_t* up_source = valid_output
        ? up_weight + output_column * packed_cols + weight_pack
        : up_weight;
    threadgroup T* gate_destination =
        gate_tile + weight_output * padded_reduction + weight_pack * values_per_pack;
    threadgroup T* up_destination =
        up_tile + weight_output * padded_reduction + weight_pack * values_per_pack;

    if (thread_index < block) {
        const int parameter_output = column_base + thread_index;
        const bool valid_parameter = parameter_output < output_dim;
        const int parameter_index = parameter_output * groups_per_output;
        parameters[thread_index] = valid_parameter ? gate_scales[parameter_index] : T(0);
        parameters[block + thread_index] = valid_parameter ? gate_biases[parameter_index] : T(0);
        parameters[2 * block + thread_index] = valid_parameter ? up_scales[parameter_index] : T(0);
        parameters[3 * block + thread_index] = valid_parameter ? up_biases[parameter_index] : T(0);
    }

    int group_step = 0;
    for (int reduction_base = 0; reduction_base < input_dim; reduction_base += reduction) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        activation_loader::load(
            x + row_base * input_dim + reduction_base,
            input_dim,
            activation_tile,
            thread_index,
            clamp(rows - row_base, 0, block),
            min(reduction, input_dim - reduction_base));

        const uint32_t gate_packed = valid_output ? *gate_source : 0;
        const uint32_t up_packed = valid_output ? *up_source : 0;
        const float gate_scale = static_cast<float>(parameters[weight_output]);
        const float gate_bias = static_cast<float>(parameters[block + weight_output]);
        const float up_scale = static_cast<float>(parameters[2 * block + weight_output]);
        const float up_bias = static_cast<float>(parameters[3 * block + weight_output]);
        #pragma clang loop unroll(full)
        for (int value = 0; value < values_per_pack; ++value) {
            gate_destination[value] = static_cast<T>(
                static_cast<float>((gate_packed >> (value * 4)) & mask) * gate_scale + gate_bias);
            up_destination[value] = static_cast<T>(
                static_cast<float>((up_packed >> (value * 4)) & mask) * up_scale + up_bias);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        gate_mma.multiply_accumulate(activation_tile, gate_tile);
        up_mma.multiply_accumulate(activation_tile, up_tile);
        gate_source += reduction / values_per_pack;
        up_source += reduction / values_per_pack;
        group_step += reduction;
        if (group_step == group_size) {
            group_step = 0;
            const int next_reduction = reduction_base + reduction;
            if (next_reduction < input_dim && thread_index < block) {
                const int parameter_output = column_base + thread_index;
                const bool valid_parameter = parameter_output < output_dim;
                const int parameter_index =
                    parameter_output * groups_per_output + next_reduction / group_size;
                parameters[thread_index] = valid_parameter ? gate_scales[parameter_index] : T(0);
                parameters[block + thread_index] = valid_parameter ? gate_biases[parameter_index] : T(0);
                parameters[2 * block + thread_index] = valid_parameter ? up_scales[parameter_index] : T(0);
                parameters[3 * block + thread_index] = valid_parameter ? up_biases[parameter_index] : T(0);
            }
        }
    }

    const int simdgroup_row = simdgroup / 2;
    const int simdgroup_column = simdgroup % 2;
    const ushort2 coordinate = tiny_llm::course_matrix_coordinate(lane);
    #pragma unroll
    for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
        const int row = simdgroup_row * 16 + row_fragment * 8 + coordinate.y;
        if (row_base + row >= rows) continue;
        #pragma unroll
        for (int column_fragment = 0; column_fragment < 2; ++column_fragment) {
            const int column = simdgroup_column * 16 + column_fragment * 8 + coordinate.x;
            #pragma unroll
            for (int element = 0; element < 2; ++element) {
                if (column_base + column + element >= output_dim) continue;
                const float gate =
                    gate_mma.accumulators[row_fragment][column_fragment].thread_elements()[element];
                const float up =
                    up_mma.accumulators[row_fragment][column_fragment].thread_elements()[element];
                out[(row_base + row) * output_dim + column_base + column + element] =
                    static_cast<T>((gate / (1.0f + fast::exp(-gate))) * up);
            }
        }
    }
}

template <typename T>
[[kernel]] void week2_decode_attention(
    device const T* q [[buffer(0)]],
    device const T* k [[buffer(1)]],
    device const T* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& q_rows [[buffer(5)]],
    constant const int& length [[buffer(6)]],
    constant const int& context [[buffer(7)]],
    constant const int& dim [[buffer(8)]],
    constant const int& num_heads [[buffer(9)]],
    constant const int& num_kv_heads [[buffer(10)]],
    constant const float& scale [[buffer(11)]],
    constant const int& is_causal [[buffer(12)]],
    constant const int& has_mask [[buffer(13)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint query_index [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (query_index >= q_rows * length) return;
    const int query_row = query_index / length;
    const int query_position = query_index % length;
    const int batch = query_row / num_heads;
    const int query_head = query_row % num_heads;
    const int kv_head = query_head / (num_heads / num_kv_heads);
    const int kv_row = batch * num_kv_heads + kv_head;

    constexpr int max_values_per_lane = 8;
    constexpr int simdgroups_per_query = 32;
    float accumulator[max_values_per_lane] = {0.0f};
    float query_values[max_values_per_lane] = {0.0f};
    float max_score = -1e30f;
    float sum = 0.0f;
    const int values_per_lane = (dim + 31) / 32;

    for (int item = 0; item < values_per_lane; ++item) {
        const int d = lane + item * 32;
        if (d < dim && item < max_values_per_lane) {
            query_values[item] =
                static_cast<float>(q[query_index * dim + d]) * scale;
        }
    }

    for (int position = simdgroup; position < context; position += simdgroups_per_query) {
        if (is_causal && position > context - length + query_position) continue;
        float partial = 0.0f;
        for (int item = 0; item < values_per_lane; ++item) {
            const int d = lane + item * 32;
            if (d < dim && item < max_values_per_lane) {
                partial += query_values[item] *
                           static_cast<float>(k[(kv_row * context + position) * dim + d]);
            }
        }
        float score = simd_sum(partial);
        if (has_mask) score += mask[query_index * context + position];
        const float new_max = max(max_score, score);
        const float old_factor = fast::exp(max_score - new_max);
        const float score_factor = fast::exp(score - new_max);
        sum = sum * old_factor + score_factor;
        for (int item = 0; item < values_per_lane; ++item) {
            const int d = lane + item * 32;
            if (d < dim && item < max_values_per_lane) {
                accumulator[item] = accumulator[item] * old_factor +
                                    score_factor * static_cast<float>(v[(kv_row * context + position) * dim + d]);
            }
        }
        max_score = new_max;
    }

    threadgroup float* partial_accumulators = scratch;
    threadgroup float* partial_maxima =
        partial_accumulators + simdgroups_per_query * dim;
    threadgroup float* partial_sums = partial_maxima + simdgroups_per_query;
    threadgroup float* partial_factors = partial_sums + simdgroups_per_query;
    if (lane == 0) {
        partial_maxima[simdgroup] = max_score;
        partial_sums[simdgroup] = sum;
    }
    for (int item = 0; item < values_per_lane; ++item) {
        const int d = lane + item * 32;
        if (d < dim && item < max_values_per_lane) {
            partial_accumulators[simdgroup * dim + d] = accumulator[item];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e30f;
    if (simdgroup == 0) {
        for (int group = 0; group < simdgroups_per_query; ++group) {
            global_max = max(global_max, partial_maxima[group]);
        }
        if (lane < simdgroups_per_query) {
            partial_factors[lane] =
                fast::exp(partial_maxima[lane] - global_max);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        float global_sum = 0.0f;
        for (int group = 0; group < simdgroups_per_query; ++group) {
            global_sum += partial_sums[group] * partial_factors[group];
        }
        partial_sums[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < dim) {
        float value_sum = 0.0f;
        for (int group = 0; group < simdgroups_per_query; ++group) {
            value_sum += partial_accumulators[group * dim + thread_index] *
                         partial_factors[group];
        }
        out[query_index * dim + thread_index] =
            static_cast<T>(value_sum / partial_sums[0]);
    }
}

// Q, K, and V consume one shared activation tile. Each projection keeps its
// own packed-W4 metadata and output width; edge guards cover all three tails.
template <typename T>
[[kernel]] void week2_quantized_qkv(
    device const T* x [[buffer(0)]],
    device const T* q_scales [[buffer(1)]],
    device const T* q_biases [[buffer(2)]],
    device const uint32_t* q_weight [[buffer(3)]],
    device const T* k_scales [[buffer(4)]],
    device const T* k_biases [[buffer(5)]],
    device const uint32_t* k_weight [[buffer(6)]],
    device const T* v_scales [[buffer(7)]],
    device const T* v_biases [[buffer(8)]],
    device const uint32_t* v_weight [[buffer(9)]],
    device T* out [[buffer(10)]],
    constant const int& rows [[buffer(11)]],
    constant const int& input_dim [[buffer(12)]],
    constant const int& q_dim [[buffer(13)]],
    constant const int& k_dim [[buffer(14)]],
    constant const int& v_dim [[buffer(15)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int block = 32;
    constexpr int reduction = 32;
    constexpr int padded_reduction = 40;
    constexpr int group_size = 128;
    constexpr int values_per_pack = 8;
    constexpr uint32_t mask = 0xf;
    const int row_base = group_id.y * block;
    const int column_base = group_id.x * block;
    const int packed_cols = input_dim / values_per_pack;
    const int groups_per_output = input_dim / group_size;
    const int total_dim = q_dim + k_dim + v_dim;

    threadgroup T activation_tile[block * padded_reduction];
    threadgroup T q_tile[block * padded_reduction];
    threadgroup T k_tile[block * padded_reduction];
    threadgroup T v_tile[block * padded_reduction];
    threadgroup T parameters[6 * block];
    using mma_type = tiny_llm::CooperativeBlockMMA<T, T, padded_reduction>;
    using activation_loader = tiny_llm::CooperativeTileLoader<
        T, block, reduction, padded_reduction, 128, false, true>;
    mma_type q_mma(simdgroup, lane);
    mma_type k_mma(simdgroup, lane);
    mma_type v_mma(simdgroup, lane);

    const int weight_output = thread_index / 4;
    const int weight_pack = thread_index % 4;
    const int output_column = column_base + weight_output;
    const bool valid_q = output_column < q_dim;
    const bool valid_k = output_column < k_dim;
    const bool valid_v = output_column < v_dim;
    device const uint32_t* q_source = valid_q ? q_weight + output_column * packed_cols + weight_pack : q_weight;
    device const uint32_t* k_source = valid_k ? k_weight + output_column * packed_cols + weight_pack : k_weight;
    device const uint32_t* v_source = valid_v ? v_weight + output_column * packed_cols + weight_pack : v_weight;
    threadgroup T* q_destination = q_tile + weight_output * padded_reduction + weight_pack * values_per_pack;
    threadgroup T* k_destination = k_tile + weight_output * padded_reduction + weight_pack * values_per_pack;
    threadgroup T* v_destination = v_tile + weight_output * padded_reduction + weight_pack * values_per_pack;

    if (thread_index < block) {
        const int parameter_output = column_base + thread_index;
        const int parameter_index = parameter_output * groups_per_output;
        const bool parameter_q = parameter_output < q_dim;
        const bool parameter_k = parameter_output < k_dim;
        const bool parameter_v = parameter_output < v_dim;
        parameters[thread_index] = parameter_q ? q_scales[parameter_index] : T(0);
        parameters[block + thread_index] = parameter_q ? q_biases[parameter_index] : T(0);
        parameters[2 * block + thread_index] = parameter_k ? k_scales[parameter_index] : T(0);
        parameters[3 * block + thread_index] = parameter_k ? k_biases[parameter_index] : T(0);
        parameters[4 * block + thread_index] = parameter_v ? v_scales[parameter_index] : T(0);
        parameters[5 * block + thread_index] = parameter_v ? v_biases[parameter_index] : T(0);
    }

    int group_step = 0;
    for (int reduction_base = 0; reduction_base < input_dim; reduction_base += reduction) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        activation_loader::load(
            x + row_base * input_dim + reduction_base,
            input_dim,
            activation_tile,
            thread_index,
            clamp(rows - row_base, 0, block),
            min(reduction, input_dim - reduction_base));

        const uint32_t q_packed = valid_q ? *q_source : 0;
        const uint32_t k_packed = valid_k ? *k_source : 0;
        const uint32_t v_packed = valid_v ? *v_source : 0;
        const float q_scale = static_cast<float>(parameters[weight_output]);
        const float q_bias = static_cast<float>(parameters[block + weight_output]);
        const float k_scale = static_cast<float>(parameters[2 * block + weight_output]);
        const float k_bias = static_cast<float>(parameters[3 * block + weight_output]);
        const float v_scale = static_cast<float>(parameters[4 * block + weight_output]);
        const float v_bias = static_cast<float>(parameters[5 * block + weight_output]);
        #pragma clang loop unroll(full)
        for (int value = 0; value < values_per_pack; ++value) {
            const int shift = value * 4;
            q_destination[value] = static_cast<T>(static_cast<float>((q_packed >> shift) & mask) * q_scale + q_bias);
            k_destination[value] = static_cast<T>(static_cast<float>((k_packed >> shift) & mask) * k_scale + k_bias);
            v_destination[value] = static_cast<T>(static_cast<float>((v_packed >> shift) & mask) * v_scale + v_bias);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        q_mma.multiply_accumulate(activation_tile, q_tile);
        k_mma.multiply_accumulate(activation_tile, k_tile);
        v_mma.multiply_accumulate(activation_tile, v_tile);
        q_source += reduction / values_per_pack;
        k_source += reduction / values_per_pack;
        v_source += reduction / values_per_pack;
        group_step += reduction;
        if (group_step == group_size) {
            group_step = 0;
            const int next_reduction = reduction_base + reduction;
            if (next_reduction < input_dim && thread_index < block) {
                const int parameter_output = column_base + thread_index;
                const int group = next_reduction / group_size;
                const bool parameter_q = parameter_output < q_dim;
                const bool parameter_k = parameter_output < k_dim;
                const bool parameter_v = parameter_output < v_dim;
                const int parameter_index = parameter_output * groups_per_output + group;
                parameters[thread_index] = parameter_q ? q_scales[parameter_index] : T(0);
                parameters[block + thread_index] = parameter_q ? q_biases[parameter_index] : T(0);
                parameters[2 * block + thread_index] = parameter_k ? k_scales[parameter_index] : T(0);
                parameters[3 * block + thread_index] = parameter_k ? k_biases[parameter_index] : T(0);
                parameters[4 * block + thread_index] = parameter_v ? v_scales[parameter_index] : T(0);
                parameters[5 * block + thread_index] = parameter_v ? v_biases[parameter_index] : T(0);
            }
        }
    }

    const int simdgroup_row = simdgroup / 2;
    const int simdgroup_column = simdgroup % 2;
    const ushort2 coordinate = tiny_llm::course_matrix_coordinate(lane);
    #pragma unroll
    for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
        const int row = simdgroup_row * 16 + row_fragment * 8 + coordinate.y;
        if (row_base + row >= rows) continue;
        #pragma unroll
        for (int column_fragment = 0; column_fragment < 2; ++column_fragment) {
            const int column = simdgroup_column * 16 + column_fragment * 8 + coordinate.x;
            #pragma unroll
            for (int element = 0; element < 2; ++element) {
                const int output = column_base + column + element;
                if (output < q_dim) {
                    out[(row_base + row) * total_dim + output] =
                        static_cast<T>(q_mma.accumulators[row_fragment][column_fragment].thread_elements()[element]);
                }
                if (output < k_dim) {
                    out[(row_base + row) * total_dim + q_dim + output] =
                        static_cast<T>(k_mma.accumulators[row_fragment][column_fragment].thread_elements()[element]);
                }
                if (output < v_dim) {
                    out[(row_base + row) * total_dim + q_dim + k_dim + output] =
                        static_cast<T>(v_mma.accumulators[row_fragment][column_fragment].thread_elements()[element]);
                }
            }
        }
    }
}

instantiate_kernel("week2_rms_norm_f32", week2_rms_norm, float);
instantiate_kernel("week2_rms_norm_f16", week2_rms_norm, half);
instantiate_kernel("week2_rms_norm_bf16", week2_rms_norm, bfloat16_t);
instantiate_kernel("week2_rope_f32", week2_rope, float);
instantiate_kernel("week2_rope_f16", week2_rope, half);
instantiate_kernel("week2_rope_bf16", week2_rope, bfloat16_t);
instantiate_kernel("week2_swiglu_f32", week2_swiglu, float);
instantiate_kernel("week2_swiglu_f16", week2_swiglu, half);
instantiate_kernel("week2_swiglu_bf16", week2_swiglu, bfloat16_t);
instantiate_kernel("week2_quantized_gate_up_swiglu_bf16", week2_quantized_gate_up_swiglu, bfloat16_t);
instantiate_kernel("week2_quantized_qkv_bf16", week2_quantized_qkv, bfloat16_t);
instantiate_kernel("week2_decode_attention_f32", week2_decode_attention, float);
instantiate_kernel("week2_decode_attention_f16", week2_decode_attention, half);
instantiate_kernel("week2_decode_attention_bf16", week2_decode_attention, bfloat16_t);
