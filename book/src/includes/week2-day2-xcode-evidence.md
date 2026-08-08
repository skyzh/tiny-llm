Reference trace: `day2-readable-q-m1.gputrace`, `tiny_llm_ref`, vanilla
packed-W4 Metal Q projection, `A[M,N] @ W[K,N]^T` with `M=1`, `N=2560`,
and `K=4096`.

Capture environment: Apple M4 Pro 20-core GPU, macOS 26.5.2, Xcode 26.6,
Metal 17.6, MLX 0.32.0, Medium performance state, Metal source revision
`44ab836`.

![Day 2 Xcode performance overview](./images/week2/xcode/week2-day2-xcode-overview.png)

![Day 2 Xcode performance limiters, left columns](./images/week2/xcode/week2-day2-xcode-limiters-left.png)

![Day 2 Xcode performance limiters, right columns](./images/week2/xcode/week2-day2-xcode-limiters-right.png)

![Day 2 Xcode memory counters, left columns](./images/week2/xcode/week2-day2-xcode-memory-left.png)

![Day 2 Xcode memory counters, right columns](./images/week2/xcode/week2-day2-xcode-memory-right.png)

![Day 2 Xcode Shader Cost Graph and weighted source lines](./images/week2/xcode/week2-day2-xcode-cost-source.png)

The full-checkpoint attribution identifies the Day 2 model's dense projections
as 81.5% of measured decode work. These screenshots show an isolated,
inspectable vanilla Metal schedule at the same Qwen shape; the readable MLX
equation remains the correctness oracle, and the trace does not substitute for
the complete-model profile.
