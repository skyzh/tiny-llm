#include <metal_stdlib>

using namespace metal;

[[kernel]] void paged_attention_f32(
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
    constant const int& Br [[buffer(15)]],
    constant const int& Bc [[buffer(16)]],
    [[maybe_unused]] constant const int& Tr [[buffer(17)]],
    constant const int& Tc [[buffer(18)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int n = group_id.x;
  const int i = group_id.y;
  const int a = simd_gid;
  const int b = simd_lid;
  const int row = i * Br + a;
  const bool is_i_in_range = n < N && row < L && a < Br;

  const int batch = n / num_heads;
  const int q_head = n % num_heads;
  const int q_kv_ratio = num_heads / num_kv_heads;
  const int kv_head = q_head / q_kv_ratio;
  const int context_len = context_lens[batch];
  const int causal_offset = context_len - L;

  threadgroup float q_local[32][128];
  threadgroup float o_i[32 * 128];

  if (simd_lid == 0) {
    for (int c = 0; c < D; c++) {
      q_local[a][c] = is_i_in_range ? q[n * L * D + row * D + c] : 0.0f;
      o_i[a * D + c] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m_i = -INFINITY;
  float l_i = 0.0f;

  for (int j = 0; j < Tc; j++) {
    const int col = j * Bc + b;
    if (j * Bc >= context_len) {
      continue;
    }
    if (is_causal) {
      const int row_max = min((i + 1) * Br - 1, L - 1);
      if (j * Bc > row_max + causal_offset) {
        continue;
      }
    }

    const int page_idx = col / page_size;
    const int slot = col - page_idx * page_size;
    int page_id = -1;
    bool is_j_in_range = col < context_len && b < Bc && page_idx < max_pages;
    if (is_j_in_range) {
      page_id = block_table[batch * max_pages + page_idx];
      is_j_in_range = page_id >= 0;
    }

    bool visible = is_i_in_range && is_j_in_range;
    if (visible && is_causal) {
      visible = col <= row + causal_offset;
    }

    float s_a_b = -INFINITY;
    if (visible) {
      float score = 0.0f;
      for (int c = 0; c < D; c++) {
        const int k_idx =
            ((page_id * num_kv_heads + kv_head) * page_size + slot) * D + c;
        score += q_local[a][c] * key_pages[k_idx];
      }
      s_a_b = score * scale;
    }

    const float rowmax = simd_max(s_a_b);
    const float new_max = max(m_i, rowmax);
    const float old_scale = exp(m_i - new_max);
    m_i = new_max;

    const float p_a_b = visible ? exp(s_a_b - m_i) : 0.0f;
    const float rowsum = simd_sum(p_a_b);
    l_i = old_scale * l_i + rowsum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = 0; c < D; c++) {
      float partial = 0.0f;
      if (visible) {
        const int v_idx =
            ((page_id * num_kv_heads + kv_head) * page_size + slot) * D + c;
        partial = p_a_b * value_pages[v_idx];
      }
      const float res = simd_sum(partial);
      if (simd_lid == 0 && is_i_in_range) {
        o_i[a * D + c] = old_scale * o_i[a * D + c] + res;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int c = 0; c < D; c++) {
      if (is_i_in_range) {
        out[n * L * D + row * D + c] =
            l_i > 0.0f ? o_i[a * D + c] / l_i : 0.0f;
      }
    }
  }
}
