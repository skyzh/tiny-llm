#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

namespace {

constant constexpr int HEAD_DIM = 128;
constant constexpr int MMA_SIZE = 8;

inline ushort2 matrix_coord(ushort lane) {
    const ushort quad = lane / 4;
    const ushort row = (quad & 4) + ((lane / 2) % 4);
    const ushort col = (quad & 2) * 2 + (lane % 2) * 2;
    return ushort2(col, row);
}

template <typename T>
inline void load_matrix(
    thread simdgroup_matrix<T, MMA_SIZE, MMA_SIZE>& matrix,
    threadgroup const T* source,
    int row_stride,
    ushort lane) {
    const ushort2 coord = matrix_coord(lane);
    matrix.thread_elements()[0] = source[coord.y * row_stride + coord.x];
    matrix.thread_elements()[1] = source[coord.y * row_stride + coord.x + 1];
}

template <int N>
inline float row_max(
    thread const simdgroup_matrix<float, MMA_SIZE, MMA_SIZE>* matrices) {
    float value = -INFINITY;
    for (int fragment_idx = 0; fragment_idx < N; fragment_idx++) {
        const auto values = matrices[fragment_idx].thread_elements();
        value = max(value, max(values[0], values[1]));
    }
    value = max(value, simd_shuffle_xor(value, ushort(1)));
    value = max(value, simd_shuffle_xor(value, ushort(8)));
    return value;
}

template <int N>
inline float row_sum(
    thread const simdgroup_matrix<float, MMA_SIZE, MMA_SIZE>* matrices) {
    float value = 0.0f;
    for (int fragment_idx = 0; fragment_idx < N; fragment_idx++) {
        const auto values = matrices[fragment_idx].thread_elements();
        value += values[0] + values[1];
    }
    value += simd_shuffle_xor(value, ushort(1));
    value += simd_shuffle_xor(value, ushort(8));
    return value;
}

inline void clear_matrix(thread simdgroup_matrix<float, MMA_SIZE, MMA_SIZE>& matrix) {
    matrix.thread_elements()[0] = 0.0f;
    matrix.thread_elements()[1] = 0.0f;
}

inline void scale_matrix_rows(
    thread simdgroup_matrix<float, MMA_SIZE, MMA_SIZE>& matrix,
    float scale) {
    matrix.thread_elements()[0] *= scale;
    matrix.thread_elements()[1] *= scale;
}

template <typename T>
inline void matrix_multiply_accumulate(
    thread simdgroup_matrix<float, MMA_SIZE, MMA_SIZE>& accumulator,
    thread simdgroup_matrix<T, MMA_SIZE, MMA_SIZE>& left,
    thread simdgroup_matrix<T, MMA_SIZE, MMA_SIZE>& right) {
    simdgroup_matrix<float, MMA_SIZE, MMA_SIZE> result;
    simdgroup_multiply_accumulate(result, left, right, accumulator);
    accumulator = result;
}

}  // namespace

template <typename T>
[[kernel]] void paged_attention_decode(
    device const T* q [[buffer(0)]],
    device const T* key_pages [[buffer(1)]],
    device const T* value_pages [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device const int* context_lens [[buffer(4)]],
    device T* out [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& L [[buffer(7)]],
    constant const int& D [[buffer(8)]],
    constant const int& page_size [[buffer(9)]],
    constant const int& max_pages [[buffer(10)]],
    constant const int& is_causal [[buffer(11)]],
    constant const int& num_kv_heads [[buffer(12)]],
    constant const int& num_heads [[buffer(13)]],
    constant const float& scale [[buffer(14)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint query_index [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (query_index >= N * L) return;

    constexpr int MAX_VALUES_PER_LANE = 4;
    constexpr int SIMD_GROUPS_PER_QUERY = 32;
    const int n = query_index / L;
    const int query_position = query_index - n * L;
    const int batch = n / num_heads;
    const int query_head = n - batch * num_heads;
    const int kv_head = query_head / (num_heads / num_kv_heads);
    const int context = context_lens[batch];
    const int values_per_lane = (D + 31) / 32;

    float accumulator[MAX_VALUES_PER_LANE] = {0.0f};
    float query_values[MAX_VALUES_PER_LANE] = {0.0f};
    float max_score = -INFINITY;
    float sum = 0.0f;

    for (int item = 0; item < values_per_lane; item++) {
        const int dim = lane + item * 32;
        if (dim < D && item < MAX_VALUES_PER_LANE) {
            query_values[item] = static_cast<float>(q[query_index * D + dim]) * scale;
        }
    }

    const int visible_context = is_causal
        ? clamp(context - L + query_position + 1, 0, context)
        : context;
    const int visible_pages = min(
        max_pages,
        (visible_context + page_size - 1) / page_size);
    for (int logical_page = 0; logical_page < visible_pages; logical_page++) {
        const int page_id = block_table[batch * max_pages + logical_page];
        if (page_id < 0) continue;
        const int page_start = logical_page * page_size;
        const int live_slots = min(page_size, visible_context - page_start);
        for (
            int slot = simdgroup;
            slot < live_slots;
            slot += SIMD_GROUPS_PER_QUERY) {
            const int page_offset =
                ((page_id * num_kv_heads + kv_head) * page_size + slot) * D;

            float partial = 0.0f;
            for (int item = 0; item < values_per_lane; item++) {
                const int dim = lane + item * 32;
                if (dim < D && item < MAX_VALUES_PER_LANE) {
                    partial += query_values[item] *
                        static_cast<float>(key_pages[page_offset + dim]);
                }
            }
            const float score = simd_sum(partial);
            const float new_max = max(max_score, score);
            const float old_factor = max_score == -INFINITY
                ? 0.0f
                : fast::exp(max_score - new_max);
            const float score_factor = fast::exp(score - new_max);
            sum = sum * old_factor + score_factor;
            for (int item = 0; item < values_per_lane; item++) {
                const int dim = lane + item * 32;
                if (dim < D && item < MAX_VALUES_PER_LANE) {
                    accumulator[item] = accumulator[item] * old_factor +
                        score_factor *
                        static_cast<float>(value_pages[page_offset + dim]);
                }
            }
            max_score = new_max;
        }
    }

    threadgroup float* partial_accumulators = scratch;
    threadgroup float* partial_maxima = partial_accumulators + SIMD_GROUPS_PER_QUERY * D;
    threadgroup float* partial_sums = partial_maxima + SIMD_GROUPS_PER_QUERY;
    threadgroup float* partial_factors = partial_sums + SIMD_GROUPS_PER_QUERY;
    if (lane == 0) {
        partial_maxima[simdgroup] = max_score;
        partial_sums[simdgroup] = sum;
    }
    for (int item = 0; item < values_per_lane; item++) {
        const int dim = lane + item * 32;
        if (dim < D && item < MAX_VALUES_PER_LANE) {
            partial_accumulators[simdgroup * D + dim] = accumulator[item];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -INFINITY;
    if (simdgroup == 0) {
        for (int group = 0; group < SIMD_GROUPS_PER_QUERY; group++) {
            global_max = max(global_max, partial_maxima[group]);
        }
        if (lane < SIMD_GROUPS_PER_QUERY) {
            partial_factors[lane] = global_max == -INFINITY
                ? 0.0f
                : fast::exp(partial_maxima[lane] - global_max);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        float global_sum = 0.0f;
        for (int group = 0; group < SIMD_GROUPS_PER_QUERY; group++) {
            global_sum += partial_sums[group] * partial_factors[group];
        }
        partial_sums[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < D) {
        float value_sum = 0.0f;
        for (int group = 0; group < SIMD_GROUPS_PER_QUERY; group++) {
            value_sum += partial_accumulators[group * D + thread_index] * partial_factors[group];
        }
        out[query_index * D + thread_index] = partial_sums[0] == 0.0f
            ? static_cast<T>(0.0f)
            : static_cast<T>(value_sum / partial_sums[0]);
    }
}

instantiate_kernel("paged_attention_decode_f32", paged_attention_decode, float);
instantiate_kernel("paged_attention_decode_bf16", paged_attention_decode, bfloat16_t);

[[kernel, max_total_threads_per_threadgroup(256)]] void paged_attention_mma_bf16_d128(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* key_pages [[buffer(1)]],
    device const bfloat* value_pages [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device const int* context_lens [[buffer(4)]],
    device bfloat* out [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& L [[buffer(7)]],
    constant const int& page_size [[buffer(8)]],
    constant const int& max_pages [[buffer(9)]],
    constant const int& is_causal [[buffer(10)]],
    constant const int& num_kv_heads [[buffer(11)]],
    constant const int& num_heads [[buffer(12)]],
    constant const float& scale [[buffer(13)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int BQ = 64;
    constexpr int BK = 32;
    constexpr int SIMD_GROUPS = 8;
    constexpr int THREADS = SIMD_GROUPS * 32;
    constexpr int LDQ = HEAD_DIM + 2;
    constexpr int LDK = BK + 2;
    constexpr int LDV = HEAD_DIM + 2;
    constexpr int KV_STORAGE = HEAD_DIM * LDK;
    constexpr int OUTPUT_FRAGMENTS = HEAD_DIM / MMA_SIZE;
    constexpr int SCORE_FRAGMENTS = BK / MMA_SIZE;

    const int query_block = group_id.x;
    const int query_head = group_id.y;
    const int batch = group_id.z;
    const int n = batch * num_heads + query_head;
    const int kv_head = query_head / (num_heads / num_kv_heads);
    const int context = context_lens[batch];
    const int thread_idx = simd_gid * 32 + lane;
    const ushort2 coord = matrix_coord(lane);
    const int query_row_in_block = simd_gid * MMA_SIZE + coord.y;
    const int query_row = query_block * BQ + query_row_in_block;
    const bool query_valid = n < N && query_row < L;

    device const bfloat* q_head = q + n * L * HEAD_DIM;
    threadgroup bfloat q_tile[BQ * LDQ];
    threadgroup bfloat kv_tile[KV_STORAGE];
    threadgroup int physical_pages[BK];

    for (int idx = thread_idx; idx < BQ * HEAD_DIM; idx += THREADS) {
        const int row = idx / HEAD_DIM;
        const int dim = idx - row * HEAD_DIM;
        const int global_row = query_block * BQ + row;
        q_tile[row * LDQ + dim] = global_row < L ? q_head[global_row * HEAD_DIM + dim] : bfloat(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, MMA_SIZE, MMA_SIZE> output[OUTPUT_FRAGMENTS];
    for (int output_fragment = 0; output_fragment < OUTPUT_FRAGMENTS; output_fragment++) {
        clear_matrix(output[output_fragment]);
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const int total_kv_tiles = (context + BK - 1) / BK;
    int kv_tile_limit = total_kv_tiles;
    if (is_causal) {
        const int last_query = min((query_block + 1) * BQ, L) - 1;
        const int last_visible_key = last_query + (context - L);
        kv_tile_limit = clamp((last_visible_key + 1 + BK - 1) / BK, 0, total_kv_tiles);
    }
    const bool single_page_tile = page_size >= BK && page_size % BK == 0;

    for (int tile_idx = 0; tile_idx < kv_tile_limit; tile_idx++) {
        const int tile_start = tile_idx * BK;
        const int tile_logical_page = tile_start / page_size;
        const int tile_page_slot = tile_start - tile_logical_page * page_size;
        if (single_page_tile) {
            if (thread_idx == 0) {
                physical_pages[0] = tile_start < context && tile_logical_page < max_pages
                    ? block_table[batch * max_pages + tile_logical_page]
                    : -1;
            }
        } else if (thread_idx < BK) {
            const int key = tile_start + thread_idx;
            const int logical_page = key / page_size;
            physical_pages[thread_idx] =
                key < context && logical_page < max_pages
                    ? block_table[batch * max_pages + logical_page]
                    : -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int idx = thread_idx; idx < BK * HEAD_DIM; idx += THREADS) {
            const int key = idx / HEAD_DIM;
            const int dim = idx - key * HEAD_DIM;
            const int global_key = tile_start + key;
            const int logical_page = global_key / page_size;
            const int slot = single_page_tile
                ? tile_page_slot + key
                : global_key - logical_page * page_size;
            const int page_id = single_page_tile
                ? physical_pages[0]
                : physical_pages[key];
            const int page_offset = ((page_id * num_kv_heads + kv_head) * page_size + slot) * HEAD_DIM;
            kv_tile[dim * LDK + key] = page_id >= 0
                ? key_pages[page_offset + dim]
                : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, MMA_SIZE, MMA_SIZE> scores[SCORE_FRAGMENTS];
        for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; key_fragment++) {
            clear_matrix(scores[key_fragment]);
        }
        for (int dim = 0; dim < HEAD_DIM; dim += MMA_SIZE) {
            simdgroup_matrix<bfloat, MMA_SIZE, MMA_SIZE> q_fragment;
            load_matrix(q_fragment, q_tile + simd_gid * MMA_SIZE * LDQ + dim, LDQ, lane);
            for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; key_fragment++) {
                simdgroup_matrix<bfloat, MMA_SIZE, MMA_SIZE> k_fragment;
                load_matrix(k_fragment, kv_tile + dim * LDK + key_fragment * MMA_SIZE, LDK, lane);
                matrix_multiply_accumulate(scores[key_fragment], q_fragment, k_fragment);
            }
        }

        for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; key_fragment++) {
            thread auto& score_values = scores[key_fragment].thread_elements();
            for (int element = 0; element < 2; element++) {
                const int tile_key = key_fragment * MMA_SIZE + coord.x + element;
                const int key = tile_start + tile_key;
                const int page_id = single_page_tile
                    ? physical_pages[0]
                    : physical_pages[tile_key];
                bool valid = query_valid && key < context && page_id >= 0;
                if (is_causal) valid = valid && key <= query_row + (context - L);
                score_values[element] = valid ? score_values[element] * scale : -INFINITY;
            }
        }

        const float tile_max = row_max<SCORE_FRAGMENTS>(scores);
        const float new_max = max(running_max, tile_max);
        const bool finite_row = query_valid && new_max != -INFINITY;
        const float previous_scale = running_max == -INFINITY || !finite_row
            ? 0.0f
            : fast::exp(running_max - new_max);
        for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; key_fragment++) {
            thread auto& score_values = scores[key_fragment].thread_elements();
            for (int element = 0; element < 2; element++) {
                score_values[element] = score_values[element] == -INFINITY || !finite_row
                    ? 0.0f
                    : fast::exp(score_values[element] - new_max);
            }
        }
        const float tile_sum = row_sum<SCORE_FRAGMENTS>(scores);
        running_max = new_max;
        running_sum = previous_scale * running_sum + tile_sum;
        for (int output_fragment = 0; output_fragment < OUTPUT_FRAGMENTS; output_fragment++) {
            scale_matrix_rows(output[output_fragment], previous_scale);
        }

        simdgroup_matrix<bfloat, MMA_SIZE, MMA_SIZE> probabilities[SCORE_FRAGMENTS];
        for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; key_fragment++) {
            const auto score_values = scores[key_fragment].thread_elements();
            probabilities[key_fragment].thread_elements()[0] = bfloat(score_values[0]);
            probabilities[key_fragment].thread_elements()[1] = bfloat(score_values[1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = thread_idx; idx < BK * HEAD_DIM; idx += THREADS) {
            const int key = idx / HEAD_DIM;
            const int dim = idx - key * HEAD_DIM;
            const int global_key = tile_start + key;
            const int logical_page = global_key / page_size;
            const int slot = single_page_tile
                ? tile_page_slot + key
                : global_key - logical_page * page_size;
            const int page_id = single_page_tile
                ? physical_pages[0]
                : physical_pages[key];
            const int page_offset = ((page_id * num_kv_heads + kv_head) * page_size + slot) * HEAD_DIM;
            kv_tile[key * LDV + dim] = page_id >= 0
                ? value_pages[page_offset + dim]
                : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int output_fragment = 0; output_fragment < OUTPUT_FRAGMENTS; output_fragment++) {
            for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; key_fragment++) {
                simdgroup_matrix<bfloat, MMA_SIZE, MMA_SIZE> v_fragment;
                load_matrix(
                    v_fragment,
                    kv_tile + key_fragment * MMA_SIZE * LDV + output_fragment * MMA_SIZE,
                    LDV,
                    lane);
                matrix_multiply_accumulate(output[output_fragment], probabilities[key_fragment], v_fragment);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query_valid) {
        for (int output_fragment = 0; output_fragment < OUTPUT_FRAGMENTS; output_fragment++) {
            const auto output_values = output[output_fragment].thread_elements();
            for (int element = 0; element < 2; element++) {
                const int dim = output_fragment * MMA_SIZE + coord.x + element;
                out[n * L * HEAD_DIM + query_row * HEAD_DIM + dim] = running_sum == 0.0f
                    ? bfloat(0.0f)
                    : bfloat(output_values[element] / running_sum);
            }
        }
    }
}

[[kernel]] void paged_attention_scalar_f32(
    device const float* q [[buffer(0)]],
    device const float* key_pages [[buffer(1)]],
    device const float* value_pages [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device const int* context_lens [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& L [[buffer(7)]],
    constant const int& D [[buffer(8)]],
    constant const int& page_size [[buffer(9)]],
    constant const int& max_pages [[buffer(10)]],
    constant const int& is_causal [[buffer(11)]],
    constant const int& num_kv_heads [[buffer(12)]],
    constant const int& num_heads [[buffer(13)]],
    constant const float& scale [[buffer(14)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int BQ = 16;
    constexpr int BK = 32;
    constexpr int MAX_D = 128;
    constexpr int OUTPUTS_PER_LANE = MAX_D / BK;
    constexpr int THREADS = BQ * BK;

    const int n = group_id.x;
    const int query_block = group_id.y;
    const int query_row_in_block = simd_gid;
    const int query_row = query_block * BQ + query_row_in_block;
    const int thread_idx = query_row_in_block * BK + lane;
    const bool query_valid = n < N && query_row < L;
    const int batch = n / num_heads;
    const int query_head = n - batch * num_heads;
    const int kv_head = query_head / (num_heads / num_kv_heads);
    const int context = context_lens[batch];

    threadgroup float q_tile[BQ * MAX_D];
    threadgroup float kv_tile[BK * MAX_D];
    threadgroup float probabilities[BQ * BK];
    threadgroup int physical_pages[BK];

    for (int idx = thread_idx; idx < BQ * MAX_D; idx += THREADS) {
        const int row = idx / MAX_D;
        const int dim = idx - row * MAX_D;
        const int global_row = query_block * BQ + row;
        q_tile[idx] = global_row < L && dim < D ? q[n * L * D + global_row * D + dim] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float output_fragment[OUTPUTS_PER_LANE] = {0.0f};
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const int total_kv_tiles = (context + BK - 1) / BK;
    int kv_tile_limit = total_kv_tiles;
    if (is_causal) {
        const int last_query = min((query_block + 1) * BQ, L) - 1;
        const int last_visible_key = last_query + (context - L);
        kv_tile_limit = clamp((last_visible_key + 1 + BK - 1) / BK, 0, total_kv_tiles);
    }
    const bool single_page_tile = page_size >= BK && page_size % BK == 0;

    for (int tile_idx = 0; tile_idx < kv_tile_limit; tile_idx++) {
        const int tile_start = tile_idx * BK;
        const int tile_logical_page = tile_start / page_size;
        const int tile_page_slot = tile_start - tile_logical_page * page_size;
        if (single_page_tile) {
            if (thread_idx == 0) {
                physical_pages[0] = tile_start < context && tile_logical_page < max_pages
                    ? block_table[batch * max_pages + tile_logical_page]
                    : -1;
            }
        } else if (thread_idx < BK) {
            const int key = tile_start + thread_idx;
            const int logical_page = key / page_size;
            physical_pages[thread_idx] =
                key < context && logical_page < max_pages
                    ? block_table[batch * max_pages + logical_page]
                    : -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = thread_idx; idx < BK * MAX_D; idx += THREADS) {
            const int key = idx / MAX_D;
            const int dim = idx - key * MAX_D;
            const int global_key = tile_start + key;
            const int logical_page = global_key / page_size;
            const int slot = single_page_tile
                ? tile_page_slot + key
                : global_key - logical_page * page_size;
            const int page_id = single_page_tile
                ? physical_pages[0]
                : physical_pages[key];
            const int page_offset = ((page_id * num_kv_heads + kv_head) * page_size + slot) * D;
            kv_tile[idx] = page_id >= 0 && dim < D ? key_pages[page_offset + dim] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int tile_key = lane;
        const int key = tile_start + tile_key;
        const int page_id = single_page_tile
            ? physical_pages[0]
            : physical_pages[tile_key];
        bool valid = query_valid && key < context && page_id >= 0;
        if (is_causal) valid = valid && key <= query_row + (context - L);
        float score = -INFINITY;
        if (valid) {
            score = 0.0f;
            for (int dim = 0; dim < D; dim++) {
                score += q_tile[query_row_in_block * MAX_D + dim] * kv_tile[tile_key * MAX_D + dim];
            }
            score *= scale;
        }

        const float tile_max = simd_max(score);
        const float new_max = max(running_max, tile_max);
        const bool finite_row = new_max != -INFINITY;
        const float previous_scale = running_max == -INFINITY || !finite_row
            ? 0.0f
            : fast::exp(running_max - new_max);
        const float probability = score == -INFINITY || !finite_row ? 0.0f : fast::exp(score - new_max);
        running_max = new_max;
        running_sum = previous_scale * running_sum + simd_sum(probability);
        probabilities[query_row_in_block * BK + lane] = probability;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = thread_idx; idx < BK * MAX_D; idx += THREADS) {
            const int value_key = idx / MAX_D;
            const int dim = idx - value_key * MAX_D;
            const int global_key = tile_start + value_key;
            const int logical_page = global_key / page_size;
            const int slot = single_page_tile
                ? tile_page_slot + value_key
                : global_key - logical_page * page_size;
            const int page_id = single_page_tile
                ? physical_pages[0]
                : physical_pages[value_key];
            const int page_offset = ((page_id * num_kv_heads + kv_head) * page_size + slot) * D;
            kv_tile[idx] = page_id >= 0 && dim < D ? value_pages[page_offset + dim] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (query_valid) {
            for (int output_idx = 0; output_idx < OUTPUTS_PER_LANE; output_idx++) {
                const int dim = lane + output_idx * BK;
                if (dim < D) {
                    float partial = 0.0f;
                    for (int value_key = 0; value_key < BK; value_key++) {
                        partial += probabilities[query_row_in_block * BK + value_key] *
                            kv_tile[value_key * MAX_D + dim];
                    }
                    output_fragment[output_idx] = previous_scale * output_fragment[output_idx] + partial;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query_valid) {
        for (int output_idx = 0; output_idx < OUTPUTS_PER_LANE; output_idx++) {
            const int dim = lane + output_idx * BK;
            if (dim < D) {
                out[n * L * D + query_row * D + dim] = running_sum == 0.0f
                    ? 0.0f
                    : output_fragment[output_idx] / running_sum;
            }
        }
    }
}
