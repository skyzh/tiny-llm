# Legacy Week 2 Day 6 URL: SIMD-Matrix Prefill

This URL preserves bookmarks from the earlier Week 2 order. SIMD-matrix
prefill is now the canonical [Day 5](./week2-05-simd-matrix-prefill.md). The
new Day 6 is an [optional workload-conditioned operator lab](./week2-06-operator-lab.md).

If you completed the former cumulative Day 6 checkpoint, keep both the SIMD
and decode-attention work. Verify that saved state with:

```bash
pdm run test --week 2 --day 6 --legacy-week2-order

pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint legacy-day-6 --model qwen3-4b
```

For the canonical sequence, verify SIMD-matrix prefill with the new Day 5 gate
and treat the retained attention implementation as the optional Day 6 branch.
New learners do not need the legacy selector.

{{#include copyright.md}}
