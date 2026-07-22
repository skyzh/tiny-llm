#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/attn.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

using namespace metal;

[[kernel]] void flash_attention_f32_e128(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int* mask_shape [[buffer(5)]],
    constant const int64_t* mask_strides [[buffer(6)]],
    device const int &is_causal [[buffer(7)]],
    device const int &N [[buffer(8)]],
    device const int &L [[buffer(9)]],
    device const int &S [[buffer(10)]],
    device const int &E [[buffer(11)]],
    device const int &num_kv_heads [[buffer(12)]],
    device const int &num_heads [[buffer(13)]],
    device const float &scale [[buffer(14)]],
    device const int &Br [[buffer(15)]],
    device const int &Bc [[buffer(16)]],
    [[maybe_unused]] device const int &Tr [[buffer(17)]],
    device const int &Tc [[buffer(18)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

    constexpr int BR = 16;
    constexpr int BC = 32;
    constexpr int MAX_E = 128;
    constexpr int OUTPUTS_PER_LANE = MAX_E / BC;
    constexpr int THREADS_PER_GROUP = BR * BC;

    const int n = group_id.x;
    const int query_block = group_id.y;
    const int query_row_in_block = simd_gid;
    const int query_row = query_block * BR + query_row_in_block;
    const int lane = simd_lid;
    const int thread_idx = query_row_in_block * BC + lane;
    const bool query_in_range = n < N && query_row < L;

    const int q_kv_ratio = num_heads / num_kv_heads;
    device const float* q_block = q + n * L * E + query_block * BR * E;
    device const float* k_head = k + (n / q_kv_ratio) * S * E;
    device const float* v_head = v + (n / q_kv_ratio) * S * E;

    // Q remains resident for the whole K/V traversal. K and V reuse the same
    // storage because P is materialized before the K tile is replaced by V.
    threadgroup float q_tile[BR * MAX_E];
    threadgroup float kv_tile[BC * MAX_E];
    threadgroup float p_tile[BR * BC];

    for (int idx = thread_idx; idx < BR * MAX_E; idx += THREADS_PER_GROUP) {
        const int row = idx / MAX_E;
        const int dim = idx - row * MAX_E;
        const bool valid = query_block * BR + row < L && dim < E;
        q_tile[idx] = valid ? q_block[row * E + dim] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float output_frag[OUTPUTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    const int total_kv_tiles = (S + BC - 1) / BC;
    int kv_tile_limit = total_kv_tiles;
    if (mask_mode == 1) {
        const int last_query = min((query_block + 1) * BR, L) - 1;
        const int last_visible_key = last_query + (S - L);
        kv_tile_limit = clamp((last_visible_key + 1 + BC - 1) / BC, 0, total_kv_tiles);
    }

    for (int tile_idx = 0; tile_idx < kv_tile_limit; tile_idx++) {
        for (int idx = thread_idx; idx < BC * MAX_E; idx += THREADS_PER_GROUP) {
            const int row = idx / MAX_E;
            const int dim = idx - row * MAX_E;
            const int key_row = tile_idx * BC + row;
            kv_tile[idx] = key_row < S && dim < E ? k_head[key_row * E + dim] : 0.0f;
        }
    }
    
    for (int j = 0; j < Tc; j++) {
        // Causal masking: if the entire block of K is masked out by causal mask, we can skip the computation for this block.
        if (is_causal) {
            int row_max = min((i + 1) * Br - 1, L - 1);
            int col_min = j * Bc;
            if (col_min > row_max + (S - L)) {
                continue;
            }
        }
       
        bool is_j_in_range = j * Bc + b < S && b < Bc;

        device const float *k_ptr = k_ptr_base + j * Bc * E;
        device const float *v_ptr = v_ptr_base + j * Bc * E;

        // compute s_i = q_i @ k_j^T; store the result of each cell in thread local memory
        float s_a_b = 0.0;
        for (int c = 0; c < E; c++) {
            if (is_i_in_range && is_j_in_range) {
                s_a_b += q_local[a][c] * k_ptr[b * E + c];
            }
        }
        s_a_b *= scale;
        if (is_i_in_range && is_j_in_range) {
            int row_min = i * Br;
            int col_max = min((j + 1) * Bc - 1, S - 1);
            bool block_all_valid = is_causal && (col_max <= row_min + (S - L));
            if (!block_all_valid) {
                int64_t m_idx_1 = n;
                int64_t m_idx_2 = i * Br + a;
                int64_t m_idx_3 = j * Bc + b;
                int64_t m_idx_converted = elem_to_loc(m_idx_1 * L * S + m_idx_2 * S + m_idx_3, mask_shape, mask_strides, 3);
                s_a_b += mask[m_idx_converted];
            }
        } else {
            s_a_b = -1e9;
        }

        // for each cell, get the rowmax of the corresponding row, and compute m_i in each
        // of the cells
        float rowmax = simd_max(s_a_b);
        float new_max = max(m_i, rowmax);
        float m_i_diff = m_i - new_max;
        float m_i_diff_exp = exp(m_i_diff);
        m_i = new_max;

        // compute matrix p_j for each of the cell
        float p_a_b;
        if (is_i_in_range && is_j_in_range) {
            p_a_b = exp(s_a_b - m_i);
        } else {
            p_a_b = 0.0;
        }

        // compute l
        // get the rowsum of each row of p_j in all of the cells
        float rowsum = simd_sum(p_a_b);
        l_i = m_i_diff_exp * l_i + rowsum;

        // compute o_i, where O is Br x E; note that this does not align
        // with the threadgroup we dispatch, so we have to do threadgroup sync
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int key_row = tile_idx * BC + lane;
        bool score_is_valid = query_in_range && key_row < S;
        if (mask_mode == 1) {
            score_is_valid = score_is_valid && key_row <= query_row + (S - L);
        }

        float score = -INFINITY;
        if (score_is_valid) {
            score = 0.0f;
            int dim = 0;
            for (; dim + 3 < E; dim += 4) {
                const int q_idx = query_row_in_block * MAX_E + dim;
                const int k_idx = lane * MAX_E + dim;
                const float4 qv = float4(q_tile[q_idx], q_tile[q_idx + 1], q_tile[q_idx + 2], q_tile[q_idx + 3]);
                const float4 kv =
                    float4(kv_tile[k_idx], kv_tile[k_idx + 1], kv_tile[k_idx + 2], kv_tile[k_idx + 3]);
                score += dot(qv, kv);
            }
            for (; dim < E; dim++) {
                score += q_tile[query_row_in_block * MAX_E + dim] * kv_tile[lane * MAX_E + dim];
            }
            score *= scale;

            if (mask_mode == 2) {
                const int64_t mask_idx =
                    elem_to_loc(n * L * S + query_row * S + key_row, mask_shape, mask_strides, 3);
                score += mask[mask_idx];
            }
        }

        const float tile_row_max = simd_max(score);
        const float new_row_max = max(row_max, tile_row_max);
        const bool has_finite_score = new_row_max != -INFINITY;
        const float previous_scale = row_max == -INFINITY ? 0.0f : fast::exp(row_max - new_row_max);
        const float probability = score == -INFINITY || !has_finite_score ? 0.0f : fast::exp(score - new_row_max);
        const float tile_row_sum = simd_sum(probability);

        row_max = new_row_max;
        row_sum = previous_scale * row_sum + tile_row_sum;
        p_tile[query_row_in_block * BC + lane] = probability;

        // All score reads from K and all writes to P must complete before the
        // shared K buffer is reused for V.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = thread_idx; idx < BC * MAX_E; idx += THREADS_PER_GROUP) {
            const int row = idx / MAX_E;
            const int dim = idx - row * MAX_E;
            const int value_row = tile_idx * BC + row;
            kv_tile[idx] = value_row < S && dim < E ? v_head[value_row * E + dim] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (query_in_range) {
            for (int frag = 0; frag < OUTPUTS_PER_LANE; frag++) {
                const int dim = lane + frag * BC;
                if (dim < E) {
                    float partial = 0.0f;
                    for (int key = 0; key < BC; key++) {
                        partial += p_tile[query_row_in_block * BC + key] * kv_tile[key * MAX_E + dim];
                    }
                    output_frag[frag] = previous_scale * output_frag[frag] + partial;
                }
            }
        }

        // No thread may replace V with the next K tile while another SIMD
        // group is still consuming it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query_in_range) {
        for (int frag = 0; frag < OUTPUTS_PER_LANE; frag++) {
            const int dim = lane + frag * BC;
            if (dim < E) {
                out[n * L * E + query_row * E + dim] = output_frag[frag] / row_sum;
            }
        }
    }
}

// The scalar kernel above is intentionally kept as a readable, general
// fallback. The production head dimension uses Metal SIMD-matrix operations
// through MLX's Steel attention building block.
instantiate_kernel(
    "flash_attention_steel_f32_bq32_bk16_bd128_wm4_wn1_maskfloat32",
    attention,
    float,
    32,
    16,
    128,
    4,
    1,
    float,
    float)
