# Legacy Week 2 Day 5 URL: Decode Attention

This URL preserves bookmarks from the earlier Week 2 order. The canonical Day
5 is now [SIMD-Matrix Prefill](./week2-05-simd-matrix-prefill.md), selected by
the preceding 128-token prefill profile. Decode attention is the supplied
branch in [Day 6's optional operator lab](./week2-06-operator-lab.md).

If you already completed the former Day 5 exercise, do not discard it. Verify
the saved checkpoint with the compatibility gate:

```bash
pdm run test --week 2 --day 5 --legacy-week2-order

pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint legacy-day-5 --model qwen3-4b
```

Then complete the new Day 5 SIMD-matrix lesson. Your additive attention work
can remain and becomes the optional Day 6 `decode-attention` branch. New
learners should follow the canonical Day 5 and Day 6 pages and do not need the
legacy selector.

{{#include copyright.md}}
