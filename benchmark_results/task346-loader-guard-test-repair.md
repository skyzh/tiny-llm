# Task 346 full-tile guard test repair

## Boundary

- Reviewed predecessor: commit
  `537ee9a7201dd3b44b022c5afdf893c7f44e92cb`, tree
  `6d6d3535562fe117cd538ef5f5e7f4062082236e`.
- Sage task #354 report attachment:
  `1d8f4eee-f8da-4f0e-a972-4365169ef162`, SHA-256
  `43cf434f7c31579a2f7a5069018afea6bcb08eb1773c803001be527c3403150e`.
- Repair scope: one new test in `tests_refsol/test_week_2_day_6.py` and this
  evidence note. No implementation or benchmark file changed.

The test compiles the real `cooperative_matrix.h` through
`mx.fast.metal_kernel`. Four threads load a full-width 4x8 BF16 source into an
exposed 4x8 destination while only two rows are valid. The first two rows hold
ordinary nonzero values; the next source row is a nonzero `37` sentinel and
the final padding row is `-11`. The test requires both valid rows to be copied
and every element in both invalid destination rows to be zero.

## Causal result

With the correct guard

```metal
valid_rows == ROWS && valid_columns == COLS
```

the probe passes. Changing only `&&` to `||` makes the same test fail: the
invalid destination rows contain the sentinel/padding values `37` and `-11`
instead of zeros. The exact correct guard was restored before all acceptance
gates and remains byte-identical to the predecessor.

## Gates

- Correct-guard causal probe: 1 passed.
- Exact `&&` to `||` mutation: 1 failed at the invalid-row zero assertion.
- Restored-guard Week 2 Day 6 suite: 18 passed, 3 model-availability skips.
- Restored-guard full reference suite after clean starter/reference builds:
  490 passed, 8 intentional model-availability skips.
- Scoped Ruff check and format check: clean.

Because the loader implementation and all 28 successor benchmark JSON files,
hash ledger, comparison report, and measured source identity are unchanged,
the accepted absolute/relative performance evidence and conservative
recommendation are unchanged.
