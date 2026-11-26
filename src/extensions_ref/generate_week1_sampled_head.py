"""Derive the Week 1 sampled fragment from the qualified MLX Steel templates."""

from hashlib import sha256
from pathlib import Path
import sys

STEEL = Path("mlx/backend/metal/kernels/steel/gemm")
PINS = {
    "kernels/steel_gemm_fused.h": "ed49c8b9c73957f0cbc9dea30f484cf84fb821aece68233d6d1967856200d5c7",
    "gemm.h": "c84af31e2c57154f2a8a24fa7f9fe2449765cce9a66f3f035846ed3c03b6a8b0",
    "mma.h": "9b37709de40c3acaa5411875e52c8d26e6de093261741f671fbdab9c39a852b7",
    "params.h": "6711e72e32310eafb088895d5a6d20fcfca814158f5b38eb72b6e69d9d773e43",
}
WRITER = r"""
template <typename T, typename AccumType, typename MMA>
METAL_FUNC void write_final_row(thread MMA& mma,
                               device T* output,
                               int local_row,
                               short rows,
                               short cols) {
  if (local_row < 0 || local_row >= rows) {
    return;
  }

  mma.apply_epilogue(TransformNone<T, AccumType>{});
  using Tile = decltype(mma.Ctile);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < Tile::kTileRows; ++i) {
    if (mma.sm + i * MMA::TM_stride == local_row) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < Tile::kTileCols; ++j) {
        const short col = mma.sn + j * MMA::TN_stride;
        const thread auto& values = mma.Ctile.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < Tile::kElemsPerFrag; ++f) {
          if (col + f < cols) {
            output[col + f] = static_cast<T>(values[f]);
          }
        }
      }
    }
  }
}
"""


def replace_checked(source: str, before: str, after: str, count: int = 1) -> str:
    if source.count(before) != count:
        raise ValueError(f"Unexpected Steel template anchor {before!r}")
    return source.replace(before, after)


def struct_template(source: str, name: str) -> str:
    struct = source.index(f"struct {name} {{")
    start = source.rindex("template <", 0, struct)
    end = source.index("\n};", struct) + len("\n};")
    return source[start:end]


def sampled_templates(source_dir: Path) -> str:
    block = struct_template((source_dir / "mma.h").read_text(), "BlockMMA")
    for before, after in (
        ("struct BlockMMA {", """struct SampledRowMMA {
  static_assert(BM == 64 && BN == 64 && BK == 16 && WM == 2 && WN == 2,
                "Sampled row must be local row 63 in the original head tile");"""),
        ("METAL_FUNC BlockMMA(", "METAL_FUNC SampledRowMMA("),
        ("  short As_offset;", "  bool owns_sampled_fragment;\n  short As_offset;"),
        (
            "    short tm = kFragSize * (simd_group_id / WN);",
            """    owns_sampled_fragment = (simd_group_id / WN == 1);
    short tm = kFragSize * (simd_group_id / WN);""",
        ),
        (
            "      tile_matmad(Ctile, Atile, Btile, Ctile);",
            """      // Local row 63 belongs to M fragment 3 of SIMD groups 2 and 3.
      // Match the original odd-row traversal and leave cooperative staging intact.
      if (owns_sampled_fragment) {
        constexpr short sampled_m = 3;
        STEEL_PRAGMA_UNROLL
        for (short n = 0; n < TN; ++n) {
          const short n_serp = TN - 1 - n;
          MMAFrag_acc_t::mma(
              Ctile.frag_at(sampled_m, n_serp),
              Atile.frag_at(sampled_m, 0),
              Btile.frag_at(0, n_serp),
              Ctile.frag_at(sampled_m, n_serp));
        }
      }""",
        ),
    ):
        block = replace_checked(block, before, after)

    tile = struct_template((source_dir / "gemm.h").read_text(), "GEMMKernel")
    tile = replace_checked(tile, "struct GEMMKernel {", "struct SampledRowGEMM {")
    tile = replace_checked(tile, "using mma_t = BlockMMA<", "using mma_t = SampledRowMMA<")
    return f"namespace mlx::steel {{\n{block}\n\n{tile}\n}}"


def generate(include_dir: Path, destination: Path) -> bool:
    source_dir = include_dir / STEEL
    for name, digest in PINS.items():
        path = source_dir / name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            return False

    original = (source_dir / "kernels/steel_gemm_fused.h").read_text()
    compact = original[original.index("// clang-format off") :]
    store = (
        "return write_final_row<T, AccumType>(mma_op, D, params->M - 1 - c_row, "
        "tgp_bm, tgp_bn);"
    )
    for before, after, count in (
        ("]] void gemm(", "]] void gemm_compact(", 1),
        ("using gemm_kernel = GEMMKernel<", "using gemm_kernel = SampledRowGEMM<", 1),
        ("D += c_row_long * params->ldd + c_col_long;", "D += c_col_long;", 1),
        ("return mma_op.store_result(D, params->ldd);", store, 2),
        (
            "return mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));",
            store,
            3,
        ),
    ):
        compact = replace_checked(compact, before, after, count)

    text = "\n".join(
        (
            '#include "mlx/backend/metal/kernels/utils.h"',
            '#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"',
            original,
            WRITER,
            sampled_templates(source_dir),
            compact,
            'instantiate_kernel("tiny_llm_week1_sampled_head", gemm_compact, '
            "bfloat16_t, 64, 64, 16, 2, 2, false, true, float);",
            "",
        )
    )
    if not destination.is_file() or destination.read_text() != text:
        destination.write_text(text)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if generate(Path(sys.argv[1]), Path(sys.argv[2])) else 2)
