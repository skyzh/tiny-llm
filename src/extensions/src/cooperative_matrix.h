#pragma once

#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

namespace tiny_llm {

// Week 2, Day 6 starter: implement this fixed-shape cooperative loader before
// using it from quantized_matmul.metal. Each thread owns one contiguous source
// chunk. Keep the full-tile path branch-free, zero-fill partial tiles, and use
// TRANSPOSE_DESTINATION only when the matrix operand needs it.
template <typename T, int ROWS, int COLS, int DESTINATION_STRIDE, int THREADS, bool TRANSPOSE_DESTINATION = false,
          bool COPY_16_BYTES = false>
struct CooperativeTileLoader {
    static_assert((ROWS * COLS) % THREADS == 0);
    static_assert(COLS % ((ROWS * COLS) / THREADS) == 0);
    static_assert(!COPY_16_BYTES || (!TRANSPOSE_DESTINATION && sizeof(T) * ((ROWS * COLS) / THREADS) == 16));

    struct alignas(sizeof(T)) Read16Bytes {
        uint8_t values[16];
    };

    static METAL_FUNC void load(device const T *source, int source_stride, threadgroup T *destination,
                                uint thread_index, int valid_rows = ROWS, int valid_columns = COLS) {
        // TODO(Week 2 Day 6): dispatch to a branch-free full-tile load or an
        // edge-safe load that writes zero outside valid_rows/valid_columns.
    }
};

// Week 2, Day 6 starter: implement the lane-to-fragment mapping and direct
// simdgroup_matrix loads used by CooperativeBlockMMA. Do not import a hidden
// matrix-multiply helper; the exercise owns this fragment bookkeeping.
METAL_FUNC ushort2 course_matrix_coordinate(ushort lane) {
    // TODO(Week 2 Day 6): map one SIMD lane to its two 8x8 fragment elements.
    return ushort2(0);
}

template <typename T>
METAL_FUNC void course_load_matrix(thread simdgroup_matrix<T, 8, 8> &matrix, threadgroup const T *source,
                                   int row_stride, ushort lane) {
    // TODO(Week 2 Day 6): load the lane's two row-major fragment elements.
}

template <typename T>
METAL_FUNC void course_load_transposed_matrix(thread simdgroup_matrix<T, 8, 8> &matrix, threadgroup const T *source,
                                              int row_stride, ushort lane) {
    // TODO(Week 2 Day 6): view row-major [output, reduction] storage as the
    // transposed right-hand matrix without materializing another tile.
}

template <typename T, typename OutT, int TILE_STRIDE>
struct CooperativeBlockMMA {
    simdgroup_matrix<float, 8, 8> accumulators[2][2];
    ushort simdgroup_index;
    ushort lane_index;

    METAL_FUNC CooperativeBlockMMA(ushort simdgroup, ushort lane) : simdgroup_index(simdgroup), lane_index(lane) {
        // TODO(Week 2 Day 6): initialize every accumulator fragment to zero.
    }

    METAL_FUNC void multiply_accumulate(threadgroup const T *left_tile, threadgroup const T *right_tile) {
        // TODO(Week 2 Day 6): load the four reduction fragments and call
        // simdgroup_multiply_accumulate for this SIMD group's 16x16 quadrant.
    }

    METAL_FUNC void store_result_safe(device OutT *output, int output_stride, short2 valid_shape) const {
        // TODO(Week 2 Day 6): store the lane-owned elements with row/column
        // guards and cast once from the FP32 accumulators.
    }
};

}  // namespace tiny_llm
