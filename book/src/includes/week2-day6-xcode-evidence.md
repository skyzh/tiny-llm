Reference trace: `day6-k-m32-simd.gputrace`, `tiny_llm_ref`, unsplit W4A16 K
projection, `A[M,N] @ W[K,N]^T` with `M=32`, `N=2560`, and `K=1024`.

Capture environment: Apple M4 Pro 20-core GPU, macOS 26.5.2, Xcode 26.6,
Metal 17.6, MLX 0.32.0, Medium performance state, Metal source revision
`44ab836`.

![Day 6 Xcode performance overview](./images/week2/xcode/week2-day6-xcode-overview.png)

![Day 6 Xcode performance limiters, left columns](./images/week2/xcode/week2-day6-xcode-limiters-left.png)

![Day 6 Xcode performance limiters, right columns](./images/week2/xcode/week2-day6-xcode-limiters-right.png)

![Day 6 Xcode memory counters, left columns](./images/week2/xcode/week2-day6-xcode-memory-left.png)

![Day 6 Xcode memory counters, right columns](./images/week2/xcode/week2-day6-xcode-memory-right.png)

![Day 6 Xcode Shader Cost Graph with the SIMD-matrix accumulation loop](./images/week2/xcode/week2-day6-xcode-cost-source.png)

The long-row controls show that the tile itself is healthy. This short K
projection launches only 32 independent result threadgroups, so its grid—not
another arithmetic rewrite—selects the Split-K experiment in Day 7.
