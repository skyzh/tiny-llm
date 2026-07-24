import argparse
import os
from pathlib import Path

import mlx.core as mx

from tiny_llm_ref.quantize import QuantizedWeights, quantized_linear


# Input and output dimensions from Qwen3-4B. Keeping this capture focused avoids
# snapshotting every model weight into the GPU trace.
PROJECTION_SHAPES = {
    "q": (2560, 4096),
    "k": (2560, 1024),
    "v": (2560, 1024),
    "o": (4096, 2560),
    "gate": (2560, 9728),
    "up": (2560, 9728),
    "down": (9728, 2560),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one Qwen3-4B Week 2 quantized projection for Xcode's "
            "Pipeline Statistics and Shader Cost Graph."
        )
    )
    parser.add_argument("--projection", choices=PROJECTION_SHAPES, default="q")
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="input rows for the vector schedule (default: 1, maximum: 8)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rows <= 8:
        parser.error("--rows must be between 1 and 8")
    if args.output.exists():
        parser.error(f"refusing to overwrite GPU trace: {args.output}")
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        parser.error("set MTL_CAPTURE_ENABLED=1 before starting the capture")
    return args


def make_projection(
    rows: int, input_dim: int, output_dim: int
) -> tuple[mx.array, QuantizedWeights]:
    group_size = 128
    bits = 4
    values_per_word = 32 // bits
    weights = QuantizedWeights(
        scales=mx.ones((output_dim, input_dim // group_size), dtype=mx.bfloat16),
        biases=mx.zeros((output_dim, input_dim // group_size), dtype=mx.bfloat16),
        group_size=group_size,
        bits=bits,
        weight=mx.zeros((output_dim, input_dim // values_per_word), dtype=mx.uint32),
    )
    x = mx.full((rows, input_dim), 2, dtype=mx.bfloat16)
    mx.eval(x, weights.weight, weights.scales, weights.biases)
    return x, weights


def main() -> None:
    args = parse_args()
    input_dim, output_dim = PROJECTION_SHAPES[args.projection]

    warmup_x, weights = make_projection(args.rows, input_dim, output_dim)
    mx.eval(quantized_linear(warmup_x, weights))

    # A different, already materialized input forces a steady-state dispatch
    # without recording allocation, random generation, or compilation work.
    capture_x = mx.full(warmup_x.shape, 3, dtype=mx.bfloat16)
    mx.eval(capture_x)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.metal.start_capture(str(args.output.resolve()))
    try:
        mx.eval(quantized_linear(capture_x, weights))
        mx.synchronize()
    finally:
        mx.metal.stop_capture()

    print(
        f"Captured Qwen3-4B {args.projection} projection "
        f"M={args.rows}, K={input_dim}, N={output_dim}: {args.output}"
    )


if __name__ == "__main__":
    main()
