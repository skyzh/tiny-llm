#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

[[kernel]] void flash_attention_f32_e128(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int* mask_shape [[buffer(5)]],
    constant const int64_t* mask_strides [[buffer(6)]],
    device const int &N [[buffer(7)]],
    device const int &L [[buffer(8)]],
    device const int &S [[buffer(9)]],
    device const int &E [[buffer(10)]],
    device const int &num_kv_heads [[buffer(11)]],
    device const int &num_heads [[buffer(12)]],
    device const float &scale [[buffer(13)]],
    device const int &Br [[buffer(14)]],
    device const int &Bc [[buffer(15)]],
    [[maybe_unused]] device const int &Tr [[buffer(16)]],
    device const int &Tc [[buffer(17)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

    int n = group_id.x;
    int i = group_id.y; // loop over Tr
    int a = simd_gid; // max=Br
    int b = simd_lid; // max=Bc

    bool is_i_in_range = i * Br + a < L && a < Br;
    const int q_kv_ratio = num_heads / num_kv_heads;
    device const float *q_ptr = q + n * L * E + i * Br * E;
    device const float *k_ptr_base = k + (n / q_kv_ratio) * S * E;
    device const float *v_ptr_base = v + (n / q_kv_ratio) * S * E;
    threadgroup float o_i[128 * 32]; // assume max(E) = 128, max(Br) = 32, only lane 0 writes to it

    if (simd_lid == 0) {
        for (int c = 0; c < E; c++) {
            o_i[a * E + c] = 0.0;
        }
    }

    threadgroup float q_local[32][128]; // assume max(E) = 128, max(Br) = 32, access by a, c
    // q_ptr: L * E
    // k_ptr: S * E
    // v_ptr: S * E
    // To access q[a, c]: use a * E + c
    // To access k/v[b, c]: use b * E + c

    float m_i = -1e9; // per thread; sync to threadgroup memory later
    float l_i = 0.0; // per thread; sync to threadgroup memory later

    // load q_local
    if (simd_lid == 0) {
        for (int c = 0; c < E; c++) {
            q_local[a][c] = q_ptr[a * E + c];
        }
    }

    if (simd_lid == 0) {
        for (int c = 0; c < E; c++) {
            if (is_i_in_range && n < N) {
                out[n * L * E + (i * Br + a) * E + c] = -233.0;
            }
        }
    }

    for (int j = 0; j < Tc; j++) {
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
            int64_t m_idx_1 = n;
            int64_t m_idx_2 = i * Br + a;
            int64_t m_idx_3 = j * Bc + b;
            int64_t m_idx_converted = elem_to_loc(m_idx_1 * L * S + m_idx_2 * S + m_idx_3, mask_shape, mask_strides, 3);
            s_a_b += mask[m_idx_converted];
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
        for (int c = 0; c < E; c++) {
            float v;
            if (is_i_in_range && is_j_in_range) {
                v = p_a_b * v_ptr[b * E + c];
            } else {
                v = 0.0;
            }
            float res = simd_sum(v); // res = sum(p_a_b * v_j) on each cell
            // only lane 0 will write to threadgroup memory
            if (simd_lid == 0 && is_i_in_range && is_j_in_range) {
                o_i[a * E + c] = m_i_diff_exp * o_i[a * E + c] + res;
            }
        }
    }

    // write to output
    if (simd_lid == 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = 0; c < E; c++) {
            if (is_i_in_range && n < N) {
                float o_i_c = o_i[a * E + c];
                o_i_c /= l_i;
                out[n * L * E + (i * Br + a) * E + c] = o_i_c;
            }
        }
    }
}
