#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

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

[[kernel, max_total_threads_per_threadgroup(256)]] void flash_attention_mma_bf16_d128(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant const int& mask_mode [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& L [[buffer(7)]],
    constant const int& S [[buffer(8)]],
    constant const int& num_kv_heads [[buffer(9)]],
    constant const int& num_heads [[buffer(10)]],
    constant const float& scale [[buffer(11)]],
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
    const int q_kv_ratio = num_heads / num_kv_heads;
    const int kv_head = query_head / q_kv_ratio;
    const int thread_idx = simd_gid * 32 + lane;
    const ushort2 coord = matrix_coord(lane);
    const int query_row_in_block = simd_gid * MMA_SIZE + coord.y;
    const int query_row = query_block * BQ + query_row_in_block;
    const bool query_valid = n < N && query_row < L;

    device const bfloat* q_head = q + n * L * HEAD_DIM;
    device const bfloat* k_head = k + (batch * num_kv_heads + kv_head) * S * HEAD_DIM;
    device const bfloat* v_head = v + (batch * num_kv_heads + kv_head) * S * HEAD_DIM;

    threadgroup bfloat q_tile[BQ * LDQ];
    threadgroup bfloat kv_tile[KV_STORAGE];

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

    const int total_kv_tiles = (S + BK - 1) / BK;
    int kv_tile_limit = total_kv_tiles;
    if (mask_mode == 1) {
        const int last_query = min((query_block + 1) * BQ, L) - 1;
        const int last_visible_key = last_query + (S - L);
        kv_tile_limit = clamp((last_visible_key + 1 + BK - 1) / BK, 0, total_kv_tiles);
    }

    for (int tile_idx = 0; tile_idx < kv_tile_limit; tile_idx++) {
        // K is transposed while loading so each MMA fragment can consume an
        // ordinary row-major [D, BK] matrix.
        for (int idx = thread_idx; idx < BK * HEAD_DIM; idx += THREADS) {
            const int key = idx / HEAD_DIM;
            const int dim = idx - key * HEAD_DIM;
            const int global_key = tile_idx * BK + key;
            kv_tile[dim * LDK + key] = global_key < S
                ? k_head[global_key * HEAD_DIM + dim]
                : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, MMA_SIZE, MMA_SIZE> scores[SCORE_FRAGMENTS];
        for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; key_fragment++) {
            clear_matrix(scores[key_fragment]);
        }

        for (int dim = 0; dim < HEAD_DIM; dim += MMA_SIZE) {
            simdgroup_matrix<bfloat, MMA_SIZE, MMA_SIZE> q_fragment;
            load_matrix(
                q_fragment,
                q_tile + simd_gid * MMA_SIZE * LDQ + dim,
                LDQ,
                lane);
            for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; key_fragment++) {
                simdgroup_matrix<bfloat, MMA_SIZE, MMA_SIZE> k_fragment;
                load_matrix(
                    k_fragment,
                    kv_tile + dim * LDK + key_fragment * MMA_SIZE,
                    LDK,
                    lane);
                matrix_multiply_accumulate(scores[key_fragment], q_fragment, k_fragment);
            }
        }

        for (int key_fragment = 0; key_fragment < SCORE_FRAGMENTS; key_fragment++) {
            thread auto& score_values = scores[key_fragment].thread_elements();
            for (int element = 0; element < 2; element++) {
                const int key = tile_idx * BK + key_fragment * MMA_SIZE + coord.x + element;
                bool valid = query_valid && key < S;
                if (mask_mode == 1) {
                    valid = valid && key <= query_row + (S - L);
                }
                if (!valid) {
                    score_values[element] = -INFINITY;
                } else {
                    score_values[element] *= scale;
                    if (mask_mode == 2) {
                        score_values[element] += mask[n * L * S + query_row * S + key];
                    }
                }
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

        // Scores now contain P. Reuse the K allocation for a row-major V tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = thread_idx; idx < BK * HEAD_DIM; idx += THREADS) {
            const int key = idx / HEAD_DIM;
            const int dim = idx - key * HEAD_DIM;
            const int global_key = tile_idx * BK + key;
            kv_tile[key * LDV + dim] = global_key < S
                ? v_head[global_key * HEAD_DIM + dim]
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
                matrix_multiply_accumulate(
                    output[output_fragment], probabilities[key_fragment], v_fragment);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query_valid && running_sum != 0.0f) {
        for (int output_fragment = 0; output_fragment < OUTPUT_FRAGMENTS; output_fragment++) {
            const auto output_values = output[output_fragment].thread_elements();
            for (int element = 0; element < 2; element++) {
                const int dim = output_fragment * MMA_SIZE + coord.x + element;
                out[n * L * HEAD_DIM + query_row * HEAD_DIM + dim] =
                    bfloat(output_values[element] / running_sum);
            }
        }
    }
}

[[kernel]] void flash_attention_scalar_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int& mask_mode [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& L [[buffer(7)]],
    constant const int& S [[buffer(8)]],
    constant const int& E [[buffer(9)]],
    constant const int& num_kv_heads [[buffer(10)]],
    constant const int& num_heads [[buffer(11)]],
    constant const float& scale [[buffer(12)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int BQ = 16;
    constexpr int BK = 32;
    constexpr int MAX_E = 128;
    constexpr int OUTPUTS_PER_LANE = MAX_E / BK;
    constexpr int THREADS = BQ * BK;

    const int n = group_id.x;
    const int query_block = group_id.y;
    const int query_row_in_block = simd_gid;
    const int query_row = query_block * BQ + query_row_in_block;
    const int thread_idx = query_row_in_block * BK + lane;
    const bool query_valid = n < N && query_row < L;
    const int q_kv_ratio = num_heads / num_kv_heads;
    device const float* q_head = q + n * L * E;
    device const float* k_head = k + (n / q_kv_ratio) * S * E;
    device const float* v_head = v + (n / q_kv_ratio) * S * E;

    threadgroup float q_tile[BQ * MAX_E];
    threadgroup float kv_tile[BK * MAX_E];
    threadgroup float probabilities[BQ * BK];

    for (int idx = thread_idx; idx < BQ * MAX_E; idx += THREADS) {
        const int row = idx / MAX_E;
        const int dim = idx - row * MAX_E;
        const int global_row = query_block * BQ + row;
        q_tile[idx] = global_row < L && dim < E ? q_head[global_row * E + dim] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float output_fragment[OUTPUTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const int total_kv_tiles = (S + BK - 1) / BK;
    int kv_tile_limit = total_kv_tiles;
    if (mask_mode == 1) {
        const int last_query = min((query_block + 1) * BQ, L) - 1;
        const int last_visible_key = last_query + (S - L);
        kv_tile_limit = clamp((last_visible_key + 1 + BK - 1) / BK, 0, total_kv_tiles);
    }

    for (int tile_idx = 0; tile_idx < kv_tile_limit; tile_idx++) {
        for (int idx = thread_idx; idx < BK * MAX_E; idx += THREADS) {
            const int key = idx / MAX_E;
            const int dim = idx - key * MAX_E;
            const int global_key = tile_idx * BK + key;
            kv_tile[idx] = global_key < S && dim < E ? k_head[global_key * E + dim] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int key = tile_idx * BK + lane;
        bool valid = query_valid && key < S;
        if (mask_mode == 1) {
            valid = valid && key <= query_row + (S - L);
        }

        float score = -INFINITY;
        if (valid) {
            score = 0.0f;
            for (int dim = 0; dim < E; dim++) {
                score += q_tile[query_row_in_block * MAX_E + dim] * kv_tile[lane * MAX_E + dim];
            }
            score *= scale;
            if (mask_mode == 2) {
                score += mask[n * L * S + query_row * S + key];
            }
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
        for (int idx = thread_idx; idx < BK * MAX_E; idx += THREADS) {
            const int value_key = idx / MAX_E;
            const int dim = idx - value_key * MAX_E;
            const int global_key = tile_idx * BK + value_key;
            kv_tile[idx] = global_key < S && dim < E ? v_head[global_key * E + dim] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (query_valid) {
            for (int output_idx = 0; output_idx < OUTPUTS_PER_LANE; output_idx++) {
                const int dim = lane + output_idx * BK;
                if (dim < E) {
                    float partial = 0.0f;
                    for (int value_key = 0; value_key < BK; value_key++) {
                        partial += probabilities[query_row_in_block * BK + value_key] *
                            kv_tile[value_key * MAX_E + dim];
                    }
                    output_fragment[output_idx] = previous_scale * output_fragment[output_idx] + partial;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query_valid && running_sum != 0.0f) {
        for (int output_idx = 0; output_idx < OUTPUTS_PER_LANE; output_idx++) {
            const int dim = lane + output_idx * BK;
            if (dim < E) {
                out[n * L * E + query_row * E + dim] = output_fragment[output_idx] / running_sum;
            }
        }
    }
}
