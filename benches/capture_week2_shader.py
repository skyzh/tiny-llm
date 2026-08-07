import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# Qwen3-4B projection dimensions as (N, K) for A[M, N] @ W[K, N]^T. Keeping
# this capture focused avoids snapshotting every model weight into the GPU trace.
PROJECTION_SHAPES = {
    "q": (2560, 4096),
    "k": (2560, 1024),
    "v": (2560, 1024),
    "o": (4096, 2560),
    "gate": (2560, 9728),
    "up": (2560, 9728),
    "down": (9728, 2560),
}

QWEN3_4B_HIDDEN_SIZE = 2560
QWEN3_4B_INTERMEDIATE_SIZE = 9728
QWEN3_4B_HEAD_DIM = 128
QWEN3_4B_NUM_ATTENTION_HEADS = 32
QWEN3_4B_RMS_NORM_EPS = 1e-6
QWEN3_4B_MAX_POSITION_EMBEDDINGS = 65_536
QWEN3_4B_ROPE_THETA = 1_000_000
POINTWISE_POSITION_OFFSET = 128

WORKLOADS = (
    "dense-projection",
    "quantized-projection",
    "pointwise",
    "decode-attention",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one Qwen3-4B Week 2 checkpoint for Xcode's Pipeline "
            "Statistics and Shader Cost Graph."
        )
    )
    parser.add_argument(
        "--workload",
        choices=WORKLOADS,
        default="quantized-projection",
        help="checkpoint workload to capture (default: quantized-projection)",
    )
    parser.add_argument(
        "--projection",
        choices=PROJECTION_SHAPES,
        default="q",
        help="projection shape for dense or quantized workloads (default: q)",
    )
    parser.add_argument(
        "--solution",
        choices=("tiny_llm", "tiny_llm_ref"),
        default="tiny_llm_ref",
        help="implementation to capture (default: tiny_llm_ref)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="input rows for dense or quantized workloads (default: 1)",
    )
    parser.add_argument(
        "--schedule",
        choices=("vanilla", "matvec", "simd-matmul", "split-k"),
        default="matvec",
        help="quantized projection schedule to capture (default: matvec)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="number of target evaluations to record (default: 10)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    if args.workload == "quantized-projection":
        if args.schedule == "matvec" and args.rows > 8:
            parser.error("the matvec schedule requires --rows <= 8")
        if args.schedule in ("simd-matmul", "split-k") and args.rows <= 8:
            parser.error("matrix schedules require --rows > 8")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.output.exists():
        parser.error(f"refusing to overwrite GPU trace: {args.output}")
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        parser.error("set MTL_CAPTURE_ENABLED=1 before starting the capture")
    return args


def make_projection(
    rows: int,
    input_dim: int,
    output_dim: int,
    weights_type: Any,
    schedule: str,
) -> tuple[mx.array, Any]:
    group_size = 128
    bits = 4
    values_per_word = 32 // bits
    weights = weights_type(
        scales=mx.ones((output_dim, input_dim // group_size), dtype=mx.bfloat16),
        biases=mx.zeros((output_dim, input_dim // group_size), dtype=mx.bfloat16),
        group_size=group_size,
        bits=bits,
        weight=mx.zeros((output_dim, input_dim // values_per_word), dtype=mx.uint32),
        use_simdgroup_matvec=schedule == "matvec",
        use_simdgroup_matmul=schedule in ("simd-matmul", "split-k"),
        use_split_k_matmul=schedule == "split-k",
    )
    x = mx.full((rows, input_dim), 2, dtype=mx.bfloat16)
    mx.eval(x, weights.weight, weights.scales, weights.biases)
    return x, weights


def prepare_quantized_projection(
    args: argparse.Namespace,
) -> tuple[str, Callable[[], list[mx.array]], Callable[[], list[mx.array]]]:
    quantize = importlib.import_module(f"{args.solution}.quantize")
    n_features, k_features = PROJECTION_SHAPES[args.projection]
    warmup_x, weights = make_projection(
        args.rows,
        n_features,
        k_features,
        quantize.QuantizedWeights,
        args.schedule,
    )
    capture_x = mx.full(warmup_x.shape, 3, dtype=mx.bfloat16)
    mx.eval(capture_x, weights.weight, weights.scales, weights.biases)

    def run(x: mx.array) -> list[mx.array]:
        return [quantize.quantized_linear(x, weights)]

    description = (
        f"{args.solution} Qwen3-4B {args.projection} projection "
        f"A[M,N] @ W[K,N]^T (M={args.rows}, N={n_features}, K={k_features}), "
        f"schedule={args.schedule}"
    )
    return description, lambda: run(warmup_x), lambda: run(capture_x)


def prepare_dense_projection(
    args: argparse.Namespace,
) -> tuple[str, Callable[[], list[mx.array]], Callable[[], list[mx.array]]]:
    basics = importlib.import_module(f"{args.solution}.basics")
    n_features, k_features = PROJECTION_SHAPES[args.projection]
    weight = mx.full((k_features, n_features), 2, dtype=mx.bfloat16)
    warmup_x = mx.full((args.rows, n_features), 2, dtype=mx.bfloat16)
    capture_x = mx.full(warmup_x.shape, 3, dtype=mx.bfloat16)
    mx.eval(weight, warmup_x, capture_x)

    def run(x: mx.array) -> list[mx.array]:
        return [basics.linear(x, weight)]

    description = (
        f"{args.solution} Qwen3-4B dense {args.projection} projection "
        f"A[M,N] @ W[K,N]^T (M={args.rows}, N={n_features}, K={k_features})"
    )
    return description, lambda: run(warmup_x), lambda: run(capture_x)


def prepare_pointwise(
    args: argparse.Namespace,
) -> tuple[str, Callable[[], list[mx.array]], Callable[[], list[mx.array]]]:
    kernels = importlib.import_module(f"{args.solution}.week2_kernels")
    weight = mx.ones((QWEN3_4B_HIDDEN_SIZE,), dtype=mx.bfloat16)
    norm = kernels.FastRMSNorm(
        QWEN3_4B_HIDDEN_SIZE,
        weight,
        eps=QWEN3_4B_RMS_NORM_EPS,
    )
    rope = kernels.FastRoPE(
        QWEN3_4B_HEAD_DIM,
        QWEN3_4B_MAX_POSITION_EMBEDDINGS,
        base=QWEN3_4B_ROPE_THETA,
    )
    offset = mx.array([POINTWISE_POSITION_OFFSET], dtype=mx.int32)

    warmup_norm = mx.full((1, 1, QWEN3_4B_HIDDEN_SIZE), 2, dtype=mx.bfloat16)
    capture_norm = mx.full(warmup_norm.shape, 3, dtype=mx.bfloat16)
    warmup_rope = mx.full(
        (1, 1, QWEN3_4B_NUM_ATTENTION_HEADS, QWEN3_4B_HEAD_DIM),
        2,
        dtype=mx.bfloat16,
    )
    capture_rope = mx.full(warmup_rope.shape, 3, dtype=mx.bfloat16)
    warmup_gate = mx.full((1, 1, QWEN3_4B_INTERMEDIATE_SIZE), 2, dtype=mx.bfloat16)
    capture_gate = mx.full(warmup_gate.shape, 3, dtype=mx.bfloat16)
    warmup_up = mx.full(warmup_gate.shape, 2, dtype=mx.bfloat16)
    capture_up = mx.full(warmup_gate.shape, 3, dtype=mx.bfloat16)
    mx.eval(
        weight,
        offset,
        warmup_norm,
        capture_norm,
        warmup_rope,
        capture_rope,
        warmup_gate,
        capture_gate,
        warmup_up,
        capture_up,
    )

    def run(
        norm_input: mx.array,
        rope_input: mx.array,
        gate: mx.array,
        up: mx.array,
    ) -> list[mx.array]:
        return [norm(norm_input), rope(rope_input, offset), kernels.swiglu(gate, up)]

    description = (
        f"{args.solution} Qwen3-4B pointwise group: "
        f"RMSNorm H={QWEN3_4B_HIDDEN_SIZE} eps={QWEN3_4B_RMS_NORM_EPS}, "
        f"RoPE H={QWEN3_4B_NUM_ATTENTION_HEADS} D={QWEN3_4B_HEAD_DIM} "
        f"offset={POINTWISE_POSITION_OFFSET} "
        f"max_position_embeddings={QWEN3_4B_MAX_POSITION_EMBEDDINGS} "
        f"theta={QWEN3_4B_ROPE_THETA}, "
        f"SwiGLU I={QWEN3_4B_INTERMEDIATE_SIZE}"
    )
    return (
        description,
        lambda: run(warmup_norm, warmup_rope, warmup_gate, warmup_up),
        lambda: run(capture_norm, capture_rope, capture_gate, capture_up),
    )


def prepare_decode_attention(
    args: argparse.Namespace,
) -> tuple[str, Callable[[], list[mx.array]], Callable[[], list[mx.array]]]:
    kernels = importlib.import_module(f"{args.solution}.week2_kernels")
    num_heads = 32
    num_kv_heads = 8
    context = 128
    head_dim = 128
    scale = head_dim**-0.5

    warmup_query = mx.full((1, num_heads, 1, head_dim), 2, dtype=mx.bfloat16)
    capture_query = mx.full(warmup_query.shape, 3, dtype=mx.bfloat16)
    key = mx.full((1, num_kv_heads, context, head_dim), 2, dtype=mx.bfloat16)
    value = mx.full(key.shape, 3, dtype=mx.bfloat16)
    mx.eval(warmup_query, capture_query, key, value)

    def run(query: mx.array) -> list[mx.array]:
        return [kernels.decode_attention_custom(query, key, value, scale)]

    description = (
        f"{args.solution} Qwen3-4B decode attention B=1, Hq={num_heads}, "
        f"Hkv={num_kv_heads}, L=1, S={context}, D={head_dim}"
    )
    return description, lambda: run(warmup_query), lambda: run(capture_query)


def prepare_workload(
    args: argparse.Namespace,
) -> tuple[str, Callable[[], list[mx.array]], Callable[[], list[mx.array]]]:
    if args.workload == "dense-projection":
        return prepare_dense_projection(args)
    if args.workload == "pointwise":
        return prepare_pointwise(args)
    if args.workload == "decode-attention":
        return prepare_decode_attention(args)
    return prepare_quantized_projection(args)


def main() -> None:
    args = parse_args()
    description, warmup, capture = prepare_workload(args)
    mx.eval(*warmup())
    # Compile the exact captured path before recording it. Each capture builder
    # still creates a fresh lazy graph over already materialized inputs.
    mx.eval(*capture())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.metal.start_capture(str(args.output.resolve()))
    try:
        for _ in range(args.iterations):
            mx.eval(*capture())
        mx.synchronize()
    finally:
        mx.metal.stop_capture()

    print(f"Captured {description}, iterations={args.iterations}: {args.output}")


if __name__ == "__main__":
    main()
