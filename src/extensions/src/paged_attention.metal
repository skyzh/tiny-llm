#include <metal_stdlib>

using namespace metal;

// Week 3, Day 3:
//   paged_cache_update_kernel
// Week 3, Day 4 reference names (the public tests do not require this split):
//   paged_attention_decode
//   paged_attention_scalar_f32
//   paged_attention_scalar_bf16
// Week 3, Day 5:
//   paged_attention_mma_bf16_d128
//
// The public C++ API remains paged_attention across Days 4-5. Day 4 may use one
// correct direct kernel for all shapes; Day 5 optimizes the supported BF16
// long-prefill region behind that stable boundary. Its behavior tests may
// already pass through the Day 4 fallback; implementing and manually tracing
// the reachable optimized region is the learner-owned Day 5 work.
