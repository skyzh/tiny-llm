#include <metal_stdlib>

using namespace metal;

// Week 3, Day 3:
//   paged_cache_update_kernel
// Week 3, Day 4:
//   paged_attention_decode
//   paged_attention_scalar_f32
// Week 3, Day 5:
//   paged_attention_mma_bf16_d128
//
// The public C++ API remains paged_attention across Days 4-5. Day 5 replaces
// only the long-query schedule behind that stable boundary.
