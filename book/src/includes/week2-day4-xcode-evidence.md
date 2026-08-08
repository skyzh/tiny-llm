Reference trace: `day4-pointwise-qwen.gputrace`, `tiny_llm_ref`, Qwen3-4B
decode rows (`B=1`, `L=1`): RMSNorm (`H=2560`, `eps=1e-6`), RoPE (`H=32`,
`D=128`, offset `128`, maximum positions `65,536`, theta `1,000,000`), and
SwiGLU (`I=9728`).

Capture environment: Apple M4 Pro 20-core GPU, macOS 26.5.2, Xcode 26.6,
Metal 17.6, MLX 0.32.0, Medium performance state, Metal source revision
`44ab836`.

![Day 4 Xcode overview with RMSNorm, RoPE, and SwiGLU pipelines](./images/week2/xcode/week2-day4-xcode-overview.png)

![Day 4 Xcode performance limiters, left columns](./images/week2/xcode/week2-day4-xcode-limiters-left.png)

![Day 4 Xcode performance limiters, right columns](./images/week2/xcode/week2-day4-xcode-limiters-right.png)

![Day 4 Xcode memory counters, left columns](./images/week2/xcode/week2-day4-xcode-memory-left.png)

![Day 4 Xcode memory counters, right columns](./images/week2/xcode/week2-day4-xcode-memory-right.png)

![Day 4 Xcode Shader Cost Graph and weighted source lines for the hottest pointwise pipeline](./images/week2/xcode/week2-day4-xcode-cost-source.png)

The overview verifies that all three intended course kernels ran. Their matched
operator and cumulative model gains shrink the pointwise cluster; the separate
context sweep then identifies attention as a removable short-context gap and
selects Day 5.
