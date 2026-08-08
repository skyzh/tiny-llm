Reference trace: `day5-decode-attention-s128.gputrace`, `tiny_llm_ref`,
Qwen3-4B grouped-query decode attention with `B=1`, `Hq=32`, `Hkv=8`, `L=1`,
`S=128`, and `D=128`.

Capture environment: Apple M4 Pro 20-core GPU, macOS 26.5.2, Xcode 26.6,
Metal 17.6, MLX 0.32.0, Medium performance state, Metal source revision
`44ab836`.

![Day 5 Xcode performance overview](./images/week2/xcode/week2-day5-xcode-overview.png)

![Day 5 Xcode performance limiters, left columns](./images/week2/xcode/week2-day5-xcode-limiters-left.png)

![Day 5 Xcode performance limiters, right columns](./images/week2/xcode/week2-day5-xcode-limiters-right.png)

![Day 5 Xcode memory counters, left columns](./images/week2/xcode/week2-day5-xcode-memory-left.png)

![Day 5 Xcode memory counters, right columns](./images/week2/xcode/week2-day5-xcode-memory-right.png)

![Day 5 Xcode Shader Cost Graph with the query-key and online-softmax loop](./images/week2/xcode/week2-day5-xcode-cost-source.png)

This is evidence for the bounded `S <= 128` schedule, not the fixed 128-token
acceptance run, whose first decode step already sees `S=129`. The matched
32-token-prompt run retains the short-context kernel. Returning to the fixed
workload makes its unchanged prefill projections the next bottleneck and
selects Day 6.
