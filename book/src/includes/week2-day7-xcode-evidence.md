Reference trace: `day7-k-m32-split-k.gputrace`, `tiny_llm_ref`, Split-K W4A16
K projection, `A[M,N] @ W[K,N]^T` with `M=32`, `N=2560`, and `K=1024`.

Capture environment: Apple M4 Pro 20-core GPU, macOS 26.5.2, Xcode 26.6,
Metal 17.6, MLX 0.32.0, Medium performance state, Metal source revision
`44ab836`.

![Day 7 Xcode overview with Split-K accumulation and reduction pipelines](./images/week2/xcode/week2-day7-xcode-overview.png)

![Day 7 Xcode performance limiters, left columns](./images/week2/xcode/week2-day7-xcode-limiters-left.png)

![Day 7 Xcode performance limiters, right columns](./images/week2/xcode/week2-day7-xcode-limiters-right.png)

![Day 7 Xcode memory counters, left columns](./images/week2/xcode/week2-day7-xcode-memory-left.png)

![Day 7 Xcode memory counters, right columns](./images/week2/xcode/week2-day7-xcode-memory-right.png)

![Day 7 Xcode Shader Cost Graph with the Split-K accumulation loop](./images/week2/xcode/week2-day7-xcode-cost-source.png)

The overview verifies both the partitioned accumulation and its merge. The
32/128/2,048-row sweep then keeps this extra work only below the measured
crossover. Week 3 changes the workload instead of adding another static
projection schedule.
