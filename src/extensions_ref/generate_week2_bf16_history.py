"""Derive BF16 history loads from the reviewed MLX GEMV templates."""

from hashlib import sha256
from pathlib import Path
import sys


GEMV = "mlx/backend/metal/kernels/gemv.h"
GEMV_SHA256 = "296ef622ea3c67d7485aa7cdf102bbfcb582e07bd3207793179a9d1e65ee5a65"


def replace(text: str, before: str, after: str, count: int = 1) -> str:
    if text.count(before) != count:
        raise ValueError("Unexpected MLX GEMV template")
    return text.replace(before, after)


def mixed_template(text: str, struct: str, entry: str, score: bool) -> str:
    name = text.index(f"struct {struct} {{")
    start = text.rindex("template <", 0, name)
    end = text.index("\n};", name) + len("\n};")
    definition = text[start:end]
    definition = replace(
        definition,
        "typename AccT = typename DefaultAccT<T>::type>",
        "typename AccT = typename DefaultAccT<T>::type, typename MatrixT = T>",
    )
    definition = replace(definition, f"struct {struct}", f"struct Mixed{struct}")
    if score:
        definition = replace(
            definition, "template <typename U = T>",
            "template <typename U = T, typename SourceT = T>", count=2,
        )
        definition = replace(definition, "const device T* src", "const device SourceT* src", count=2)
    definition = replace(
        definition, "const device T* mat [[buffer(0)]]", "const device MatrixT* mat [[buffer(0)]]",
    )

    kernel = text.index(f"[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]] void {entry}(")
    begin = text.rindex("template <", end, kernel)
    stop = text.index("\n}\n", kernel) + len("\n}")
    wrapper = text[begin:stop]
    wrapper = replace(wrapper, "    typename T,", "    typename T,\n    typename MatrixT,")
    wrapper = replace(wrapper, f"void {entry}(", f"void mixed_{entry}(")
    wrapper = replace(
        wrapper, "const device T* mat [[buffer(0)]]", "const device MatrixT* mat [[buffer(0)]]",
    )
    wrapper = replace(
        wrapper, f"{struct}<T, BM, BN, SM, SN, TM, TN, kDoAxpby>",
        f"Mixed{struct}<T, BM, BN, SM, SN, TM, TN, kDoAxpby, float, MatrixT>",
    )
    return f"{definition}\n\n{wrapper}"


def generate(include_dir: Path, destination: Path) -> bool:
    header = include_dir / GEMV
    if not header.is_file():
        return False
    data = header.read_bytes()
    if sha256(data).hexdigest() != GEMV_SHA256:
        return False

    original = data.decode()
    source = "\n\n".join((
        f'#include "{GEMV}"',
        mixed_template(original, "GEMVKernel", "gemv", score=True),
        mixed_template(original, "GEMVTKernel", "gemv_t", score=False),
        'instantiate_kernel("tiny_llm_week2_history_scores", mixed_gemv, '
        'float, bfloat16_t, 4, 1, 1, 32, 4, 4, true, false)',
        'instantiate_kernel("tiny_llm_week2_history_values", mixed_gemv_t, '
        'float, bfloat16_t, 1, 2, 8, 4, 4, 4, true, false)',
        "",
    ))
    if not destination.is_file() or destination.read_text() != source:
        destination.write_text(source)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if generate(Path(sys.argv[1]), Path(sys.argv[2])) else 2)
