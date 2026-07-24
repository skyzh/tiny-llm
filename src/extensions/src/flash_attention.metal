#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

[[kernel]] void flash_attention_f32_e128(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device const float *mask [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant const int *mask_shape [[buffer(5)]],
    constant const int64_t *mask_strides [[buffer(6)]],
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
    device const int &Tr [[buffer(17)]],
    device const int &Tc [[buffer(18)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    // TODO(student): implement flash attention kernel.
    (void)q;
    (void)k;
    (void)v;
    (void)mask;
    (void)out;
    (void)mask_shape;
    (void)mask_strides;
    (void)is_causal;
    (void)N;
    (void)L;
    (void)S;
    (void)E;
    (void)num_kv_heads;
    (void)num_heads;
    (void)scale;
    (void)Br;
    (void)Bc;
    (void)Tr;
    (void)Tc;
    (void)group_id;
    (void)simd_gid;
    (void)simd_lid;
}
