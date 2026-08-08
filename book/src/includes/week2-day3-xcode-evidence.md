Reference trace: `day3-packed-q-m1.gputrace`, `tiny_llm_ref`, packed W4A16 Q
projection, `A[M,N] @ W[K,N]^T` with `M=1`, `N=2560`, and `K=4096`.

Capture environment: Apple M4 Pro 20-core GPU, macOS 26.5.2, Xcode 26.6,
Metal 17.6, MLX 0.32.0, Medium performance state, Metal source revision
`44ab836`.

![Day 3 Xcode performance overview](./images/week2/xcode/week2-day3-xcode-overview.png)

![Day 3 Xcode performance limiters, left columns](./images/week2/xcode/week2-day3-xcode-limiters-left.png)

![Day 3 Xcode performance limiters, right columns](./images/week2/xcode/week2-day3-xcode-limiters-right.png)

![Day 3 Xcode memory counters, left columns](./images/week2/xcode/week2-day3-xcode-memory-left.png)

![Day 3 Xcode memory counters, right columns](./images/week2/xcode/week2-day3-xcode-memory-right.png)

![Day 3 Xcode Shader Cost Graph and weighted matvec source lines](./images/week2/xcode/week2-day3-xcode-cost-source.png)

The Cost Graph keeps optional instruction-level matvec headroom visible. The
matched operator table is already close to MLX, however, while the updated
full-model attribution moves the largest removable gap to the pointwise
cluster. That evidence selects Day 4.
