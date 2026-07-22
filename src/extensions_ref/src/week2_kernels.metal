#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

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

instantiate_kernel("week2_rms_norm_f32", week2_rms_norm, float);
instantiate_kernel("week2_rms_norm_f16", week2_rms_norm, half);
instantiate_kernel("week2_rms_norm_bf16", week2_rms_norm, bfloat16_t);
instantiate_kernel("week2_rope_f32", week2_rope, float);
instantiate_kernel("week2_rope_f16", week2_rope, half);
instantiate_kernel("week2_rope_bf16", week2_rope, bfloat16_t);
instantiate_kernel("week2_swiglu_f32", week2_swiglu, float);
instantiate_kernel("week2_swiglu_f16", week2_swiglu, half);
instantiate_kernel("week2_swiglu_bf16", week2_swiglu, bfloat16_t);
instantiate_kernel("week2_decode_attention_f32", week2_decode_attention, float);
instantiate_kernel("week2_decode_attention_f16", week2_decode_attention, half);
instantiate_kernel("week2_decode_attention_bf16", week2_decode_attention, bfloat16_t);
