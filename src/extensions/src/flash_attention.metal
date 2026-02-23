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


    
}
