#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/attn/loader.h"
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
[[kernel]] void paged_cache_update_kernel(
    device const T* values [[buffer(0)]],
    device T* pages [[buffer(1)]],
    constant const int& heads [[buffer(2)]],
    constant const int& length [[buffer(3)]],
    constant const int& head_dim [[buffer(4)]],
    constant const int& page_size [[buffer(5)]],
    constant const int& page_id [[buffer(6)]],
    constant const int& start [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const int total = heads * length * head_dim;
    if (index >= static_cast<uint>(total)) return;
    const int head_stride = length * head_dim;
    const int head = index / head_stride;
    const int within_head = index - head * head_stride;
    const int token = within_head / head_dim;
    const int dim = within_head - token * head_dim;
    const int destination =
        ((page_id * heads + head) * page_size + start + token) * head_dim + dim;
    pages[destination] = values[index];
}

instantiate_kernel("paged_cache_update_f32", paged_cache_update_kernel, float);
instantiate_kernel("paged_cache_update_bf16", paged_cache_update_kernel, bfloat16_t);

template <typename T, int FIXED_D = 0>
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
    uint lane [[thread_index_in_simdgroup]]) {
    if (query_index >= N * L) return;

    constexpr int MAX_VALUES_PER_LANE = 4;
    constexpr int SIMD_GROUPS_PER_QUERY = 32;
    constexpr float LOG2_E = 1.44269504089f;
    const float scale_log2 = scale * LOG2_E;
    const int dimension = FIXED_D == 0 ? D : FIXED_D;
    const int n = query_index / L;
    const int query_position = query_index - n * L;
    const int batch = n / num_heads;
    const int query_head = n - batch * num_heads;
    const int kv_head = query_head / (num_heads / num_kv_heads);
    const int context = context_lens[batch];
    const int values_per_lane = (dimension + 31) / 32;

    float accumulator[MAX_VALUES_PER_LANE] = {0.0f};
    float query_values[MAX_VALUES_PER_LANE] = {0.0f};
    float max_score = -1e30f;
    float sum = 0.0f;

    #pragma clang loop unroll(full)
    for (int item = 0; item < values_per_lane; item++) {
        const int dim = lane * values_per_lane + item;
        if (dim < dimension && item < MAX_VALUES_PER_LANE) {
            query_values[item] =
                static_cast<float>(q[query_index * D + dim]) * scale_log2;
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
            #pragma clang loop unroll(full)
            for (int item = 0; item < values_per_lane; item++) {
                const int dim = lane * values_per_lane + item;
                if (dim < dimension && item < MAX_VALUES_PER_LANE) {
                    partial += query_values[item] *
                        static_cast<float>(key_pages[page_offset + dim]);
                }
            }
            const float score = simd_sum(partial);
            const float new_max = max(max_score, score);
            const float old_factor = fast::exp2(max_score - new_max);
            const float score_factor = fast::exp2(score - new_max);
            sum = sum * old_factor + score_factor;
            #pragma clang loop unroll(full)
            for (int item = 0; item < values_per_lane; item++) {
                const int dim = lane * values_per_lane + item;
                if (dim < dimension && item < MAX_VALUES_PER_LANE) {
                    accumulator[item] = accumulator[item] * old_factor +
                        score_factor *
                        static_cast<float>(value_pages[page_offset + dim]);
                }
            }
            max_score = new_max;
        }
    }

    // Transpose the 32 partial output vectors through a compact 32x32 tile.
    // Each SIMD group then reduces one output lane with simd_sum instead of
    // assigning only D scalar threads to loop over all partials.
    threadgroup float* partial_outputs = scratch;
    threadgroup float* partial_maxima =
        partial_outputs + SIMD_GROUPS_PER_QUERY * 32;
    threadgroup float* partial_sums =
        partial_maxima + SIMD_GROUPS_PER_QUERY;
    if (lane == 0) {
        partial_maxima[simdgroup] = max_score;
        partial_sums[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial_max = partial_maxima[lane];
    const float global_max = simd_max(partial_max);
    const float factor = fast::exp2(partial_max - global_max);
    const float global_sum = simd_sum(partial_sums[lane] * factor);

    #pragma clang loop unroll(full)
    for (int item = 0; item < values_per_lane; item++) {
        partial_outputs[lane * SIMD_GROUPS_PER_QUERY + simdgroup] =
            accumulator[item];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        accumulator[item] = simd_sum(
            partial_outputs[simdgroup * SIMD_GROUPS_PER_QUERY + lane] *
            factor);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int item = 0; item < values_per_lane; item++) {
            const int dim = simdgroup * values_per_lane + item;
            if (dim < dimension && item < MAX_VALUES_PER_LANE) {
                out[query_index * D + dim] = global_sum == 0.0f
                    ? static_cast<T>(0.0f)
                    : static_cast<T>(accumulator[item] / global_sum);
            }
        }
    }
}

instantiate_kernel("paged_attention_decode_f32", paged_attention_decode, float);
instantiate_kernel("paged_attention_decode_bf16", paged_attention_decode, bfloat16_t);
instantiate_kernel("paged_attention_decode_bf16_d128", paged_attention_decode, bfloat16_t, 128);

// Qwen 4B/8B use four query heads for every KV head. Process that GQA group in
// one threadgroup so each K/V value is loaded once and reused by four queries.
[[kernel]] void paged_attention_decode_bf16_d128_gqa4(
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
    threadgroup float* scratch [[threadgroup(0)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int QUERY_HEADS_PER_KV = 4;
    constexpr int SIMD_GROUPS = 32;
    constexpr int VALUES_PER_LANE = 4;
    constexpr float LOG2_E = 1.44269504089f;
    const float scale_log2 = scale * LOG2_E;

    const int batch_size = N / num_heads;
    const int groups_per_batch = num_kv_heads * L;
    if (group_index >= batch_size * groups_per_batch) return;
    const int batch = group_index / groups_per_batch;
    const int within_batch = group_index - batch * groups_per_batch;
    const int kv_head = within_batch / L;
    const int query_position = within_batch - kv_head * L;
    const int query_head_base = kv_head * QUERY_HEADS_PER_KV;
    const int context = context_lens[batch];

    float accumulators[QUERY_HEADS_PER_KV][VALUES_PER_LANE] = {{0.0f}};
    float query_values[QUERY_HEADS_PER_KV][VALUES_PER_LANE];
    float max_scores[QUERY_HEADS_PER_KV] = {
        -1e30f, -1e30f, -1e30f, -1e30f};
    float sums[QUERY_HEADS_PER_KV] = {0.0f};

    #pragma clang loop unroll(full)
    for (int query = 0; query < QUERY_HEADS_PER_KV; query++) {
        const int query_index =
            ((batch * num_heads + query_head_base + query) * L +
             query_position);
        #pragma clang loop unroll(full)
        for (int item = 0; item < VALUES_PER_LANE; item++) {
            const int dim = lane * VALUES_PER_LANE + item;
            query_values[query][item] =
                static_cast<float>(q[query_index * HEAD_DIM + dim]) *
                scale_log2;
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
        for (int slot = simdgroup; slot < live_slots; slot += SIMD_GROUPS) {
            const int page_offset =
                ((page_id * num_kv_heads + kv_head) * page_size + slot) *
                HEAD_DIM;
            float partials[QUERY_HEADS_PER_KV] = {0.0f};
            float values[VALUES_PER_LANE];
            #pragma clang loop unroll(full)
            for (int item = 0; item < VALUES_PER_LANE; item++) {
                const int dim = lane * VALUES_PER_LANE + item;
                const float key =
                    static_cast<float>(key_pages[page_offset + dim]);
                values[item] =
                    static_cast<float>(value_pages[page_offset + dim]);
                #pragma clang loop unroll(full)
                for (int query = 0; query < QUERY_HEADS_PER_KV; query++) {
                    partials[query] += query_values[query][item] * key;
                }
            }

            #pragma clang loop unroll(full)
            for (int query = 0; query < QUERY_HEADS_PER_KV; query++) {
                const float score = simd_sum(partials[query]);
                const float new_max = max(max_scores[query], score);
                const float old_factor =
                    fast::exp2(max_scores[query] - new_max);
                const float score_factor = fast::exp2(score - new_max);
                sums[query] = sums[query] * old_factor + score_factor;
                #pragma clang loop unroll(full)
                for (int item = 0; item < VALUES_PER_LANE; item++) {
                    accumulators[query][item] =
                        accumulators[query][item] * old_factor +
                        score_factor * values[item];
                }
                max_scores[query] = new_max;
            }
        }
    }

    threadgroup float* partial_outputs = scratch;
    threadgroup float* partial_maxima =
        partial_outputs + QUERY_HEADS_PER_KV * SIMD_GROUPS * 32;
    threadgroup float* partial_sums =
        partial_maxima + QUERY_HEADS_PER_KV * SIMD_GROUPS;
    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int query = 0; query < QUERY_HEADS_PER_KV; query++) {
            partial_maxima[query * SIMD_GROUPS + simdgroup] =
                max_scores[query];
            partial_sums[query * SIMD_GROUPS + simdgroup] = sums[query];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (int query = 0; query < QUERY_HEADS_PER_KV; query++) {
        const float partial_max =
            partial_maxima[query * SIMD_GROUPS + lane];
        const float global_max = simd_max(partial_max);
        const float factor = fast::exp2(partial_max - global_max);
        const float global_sum = simd_sum(
            partial_sums[query * SIMD_GROUPS + lane] * factor);
        threadgroup float* query_partials =
            partial_outputs + query * SIMD_GROUPS * 32;

        #pragma clang loop unroll(full)
        for (int item = 0; item < VALUES_PER_LANE; item++) {
            query_partials[lane * SIMD_GROUPS + simdgroup] =
                accumulators[query][item];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            accumulators[query][item] = simd_sum(
                query_partials[simdgroup * SIMD_GROUPS + lane] * factor);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            const int query_index =
                ((batch * num_heads + query_head_base + query) * L +
                 query_position);
            #pragma clang loop unroll(full)
            for (int item = 0; item < VALUES_PER_LANE; item++) {
                const int dim = simdgroup * VALUES_PER_LANE + item;
                out[query_index * HEAD_DIM + dim] = global_sum == 0.0f
                    ? bfloat(0.0f)
                    : static_cast<bfloat>(
                        accumulators[query][item] / global_sum);
            }
        }
    }
}

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
    constexpr float LOG2_E = 1.44269504089f;
    using QBlockLoader = mlx::steel::BlockLoaderT<
        bfloat, BQ, HEAD_DIM, LDQ, 1, 1, THREADS>;
    using KBlockLoader = mlx::steel::BlockLoaderT<
        bfloat, BK, HEAD_DIM, 1, LDK, 0, THREADS>;
    using VBlockLoader = mlx::steel::BlockLoaderT<
        bfloat, BK, HEAD_DIM, LDV, 1, 0, THREADS>;

    const int query_block = group_id.x;
    const int query_head = group_id.y;
    const int batch = group_id.z;
    const int n = batch * num_heads + query_head;
    const int kv_head = query_head / (num_heads / num_kv_heads);
    const int context = context_lens[batch];
    const float scale_log2 = scale * LOG2_E;
    const int thread_idx = simd_gid * 32 + lane;
    const ushort2 coord = matrix_coord(lane);
    const int query_row_in_block = simd_gid * MMA_SIZE + coord.y;
    const int query_row = query_block * BQ + query_row_in_block;
    const bool query_valid = n < N && query_row < L;

    device const bfloat* q_head = q + n * L * HEAD_DIM;
    threadgroup bfloat q_tile[BQ * LDQ];
    threadgroup bfloat kv_tile[KV_STORAGE];
    threadgroup int physical_pages[BK];

    QBlockLoader q_loader(
        q_head + query_block * BQ * HEAD_DIM,
        HEAD_DIM,
        q_tile,
        simd_gid,
        lane);
    const int live_queries = clamp(L - query_block * BQ, 0, BQ);
    if (live_queries == BQ) {
        q_loader.load_unsafe();
    } else {
        q_loader.load_safe(short2(HEAD_DIM, live_queries));
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

        if (single_page_tile && physical_pages[0] >= 0) {
            const int page_offset =
                ((physical_pages[0] * num_kv_heads + kv_head) * page_size +
                 tile_page_slot) * HEAD_DIM;
            KBlockLoader k_loader(
                key_pages + page_offset,
                HEAD_DIM,
                kv_tile,
                simd_gid,
                lane);
            const int live_keys = clamp(context - tile_start, 0, BK);
            if (live_keys == BK) {
                k_loader.load_unsafe();
            } else {
                k_loader.load_safe(short2(HEAD_DIM, live_keys));
            }
        } else {
            for (int idx = thread_idx; idx < BK * HEAD_DIM; idx += THREADS) {
                const int key = idx / HEAD_DIM;
                const int dim = idx - key * HEAD_DIM;
                const int global_key = tile_start + key;
                const int logical_page = global_key / page_size;
                const int slot = global_key - logical_page * page_size;
                const int page_id = physical_pages[key];
                const int page_offset =
                    ((page_id * num_kv_heads + kv_head) * page_size + slot) *
                    HEAD_DIM;
                kv_tile[dim * LDK + key] = page_id >= 0
                    ? key_pages[page_offset + dim]
                    : bfloat(0.0f);
            }
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
                score_values[element] = valid
                    ? score_values[element] * scale_log2
                    : -INFINITY;
            }
        }

        const float tile_max = row_max<SCORE_FRAGMENTS>(scores);
        const float new_max = max(running_max, tile_max);
        const bool finite_row = query_valid && new_max != -INFINITY;
        const float previous_scale = running_max == -INFINITY || !finite_row
            ? 0.0f
            : fast::exp2(running_max - new_max);
        for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; key_fragment++) {
            thread auto& score_values = scores[key_fragment].thread_elements();
            for (int element = 0; element < 2; element++) {
                score_values[element] = score_values[element] == -INFINITY || !finite_row
                    ? 0.0f
                    : fast::exp2(score_values[element] - new_max);
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
        if (single_page_tile && physical_pages[0] >= 0) {
            const int page_offset =
                ((physical_pages[0] * num_kv_heads + kv_head) * page_size +
                 tile_page_slot) * HEAD_DIM;
            VBlockLoader v_loader(
                value_pages + page_offset,
                HEAD_DIM,
                kv_tile,
                simd_gid,
                lane);
            const int live_keys = clamp(context - tile_start, 0, BK);
            if (live_keys == BK) {
                v_loader.load_unsafe();
            } else {
                v_loader.load_safe(short2(HEAD_DIM, live_keys));
            }
        } else {
            for (int idx = thread_idx; idx < BK * HEAD_DIM; idx += THREADS) {
                const int key = idx / HEAD_DIM;
                const int dim = idx - key * HEAD_DIM;
                const int global_key = tile_start + key;
                const int logical_page = global_key / page_size;
                const int slot = global_key - logical_page * page_size;
                const int page_id = physical_pages[key];
                const int page_offset =
                    ((page_id * num_kv_heads + kv_head) * page_size + slot) *
                    HEAD_DIM;
                kv_tile[key * LDV + dim] = page_id >= 0
                    ? value_pages[page_offset + dim]
                    : bfloat(0.0f);
            }
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
