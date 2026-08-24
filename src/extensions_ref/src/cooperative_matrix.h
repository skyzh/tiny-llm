#pragma once

#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

namespace tiny_llm {

// Copy one fixed-shape row-major device tile into padded threadgroup storage.
// Each thread owns one contiguous source chunk.  Partial rows and columns are
// written as zero so every later matrix load can use the full logical tile.
template <
    typename T,
    int ROWS,
    int COLS,
    int DESTINATION_STRIDE,
    int THREADS,
    bool TRANSPOSE_DESTINATION = false,
    bool COPY_16_BYTES = false>
struct CooperativeTileLoader {
    static_assert((ROWS * COLS) % THREADS == 0);
    static_assert(COLS % ((ROWS * COLS) / THREADS) == 0);
    static_assert(
        !COPY_16_BYTES ||
        (!TRANSPOSE_DESTINATION &&
         sizeof(T) * ((ROWS * COLS) / THREADS) == 16));

    struct alignas(sizeof(T)) Read16Bytes {
        uint8_t values[16];
    };

    static METAL_FUNC void load(
        device const T* source,
        int source_stride,
        threadgroup T* destination,
        uint thread_index,
        int valid_rows = ROWS,
        int valid_columns = COLS) {
        if (valid_rows == ROWS && valid_columns == COLS) {
            load_full(source, source_stride, destination, thread_index);
        } else {
            load_safe(
                source,
                source_stride,
                destination,
                thread_index,
                valid_rows,
                valid_columns);
        }
    }

    static METAL_FUNC void load_full(
        device const T* source,
        int source_stride,
        threadgroup T* destination,
        uint thread_index) {
        constexpr int values_per_thread = (ROWS * COLS) / THREADS;
        constexpr int threads_per_row = COLS / values_per_thread;
        const int row = int(thread_index) / threads_per_row;
        const int column =
            (int(thread_index) % threads_per_row) * values_per_thread;

        if constexpr (COPY_16_BYTES) {
            *((threadgroup Read16Bytes*)(
                destination + row * DESTINATION_STRIDE + column)) =
                *((const device Read16Bytes*)(
                    source + row * source_stride + column));
            return;
        }

        #pragma unroll
        for (int offset = 0; offset < values_per_thread; ++offset) {
            const T value = source[row * source_stride + column + offset];
            if constexpr (TRANSPOSE_DESTINATION) {
                destination[(column + offset) * DESTINATION_STRIDE + row] = value;
            } else {
                destination[row * DESTINATION_STRIDE + column + offset] = value;
            }
        }
    }

    static METAL_FUNC void load_safe(
        device const T* source,
        int source_stride,
        threadgroup T* destination,
        uint thread_index,
        int valid_rows,
        int valid_columns) {
        constexpr int values_per_thread = (ROWS * COLS) / THREADS;
        constexpr int threads_per_row = COLS / values_per_thread;
        const int row = int(thread_index) / threads_per_row;
        const int column =
            (int(thread_index) % threads_per_row) * values_per_thread;

        #pragma unroll
        for (int offset = 0; offset < values_per_thread; ++offset) {
            const int current_column = column + offset;
            const T value = row < valid_rows && current_column < valid_columns
                ? source[row * source_stride + current_column]
                : T(0);
            if constexpr (TRANSPOSE_DESTINATION) {
                destination[current_column * DESTINATION_STRIDE + row] = value;
            } else {
                destination[row * DESTINATION_STRIDE + current_column] = value;
            }
        }
    }
};

METAL_FUNC ushort2 course_matrix_coordinate(ushort lane) {
    return ushort2(
        (lane & 1) * 2 + (lane & 8) / 2,
        (lane & 7) / 2 + (lane & 16) / 4);
}

template <typename T>
METAL_FUNC void course_load_matrix(
    thread simdgroup_matrix<T, 8, 8>& matrix,
    threadgroup const T* source,
    int row_stride,
    ushort lane) {
    const ushort2 coordinate = course_matrix_coordinate(lane);
    matrix.thread_elements()[0] = source[coordinate.y * row_stride + coordinate.x];
    matrix.thread_elements()[1] =
        source[coordinate.y * row_stride + coordinate.x + 1];
}

// Load an 8x8 fragment while viewing row-major [output, reduction] storage as
// the transposed [reduction, output] right-hand matrix.
template <typename T>
METAL_FUNC void course_load_transposed_matrix(
    thread simdgroup_matrix<T, 8, 8>& matrix,
    threadgroup const T* source,
    int row_stride,
    ushort lane) {
    const ushort2 coordinate = course_matrix_coordinate(lane);
    matrix.thread_elements()[0] = source[coordinate.x * row_stride + coordinate.y];
    matrix.thread_elements()[1] =
        source[(coordinate.x + 1) * row_stride + coordinate.y];
}

// Four simdgroups cover the four 16x16 quadrants of one 32x32 output tile.
// The wrapper only exposes the two operations the course kernel needs: add a
// 32-wide reduction tile, then store the accumulated result with edge guards.
template <typename T, typename OutT, int TILE_STRIDE>
struct CooperativeBlockMMA {
    simdgroup_matrix<float, 8, 8> accumulators[2][2];
    ushort simdgroup_index;
    ushort lane_index;

    METAL_FUNC CooperativeBlockMMA(ushort simdgroup, ushort lane)
        : simdgroup_index(simdgroup), lane_index(lane) {
        #pragma unroll
        for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
            #pragma unroll
            for (int column_fragment = 0; column_fragment < 2; ++column_fragment) {
                accumulators[row_fragment][column_fragment].thread_elements() = 0.0f;
            }
        }
    }

    METAL_FUNC void multiply_accumulate(
        threadgroup const T* left_tile,
        threadgroup const T* right_tile) {
        const int simdgroup_row = simdgroup_index / 2;
        const int simdgroup_column = simdgroup_index % 2;

        #pragma unroll
        for (int reduction_fragment = 0; reduction_fragment < 4;
             ++reduction_fragment) {
            simdgroup_matrix<T, 8, 8> left[2];
            simdgroup_matrix<T, 8, 8> right[2];

            #pragma unroll
            for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
                const int left_row = simdgroup_row * 16 + row_fragment * 8;
                course_load_matrix(
                    left[row_fragment],
                    left_tile + left_row * TILE_STRIDE + reduction_fragment * 8,
                    TILE_STRIDE,
                    lane_index);
            }

            #pragma unroll
            for (int column_fragment = 0; column_fragment < 2;
                 ++column_fragment) {
                const int right_row =
                    simdgroup_column * 16 + column_fragment * 8;
                course_load_transposed_matrix(
                    right[column_fragment],
                    right_tile + right_row * TILE_STRIDE + reduction_fragment * 8,
                    TILE_STRIDE,
                    lane_index);
            }

            #pragma unroll
            for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
                #pragma unroll
                for (int column_fragment = 0; column_fragment < 2;
                     ++column_fragment) {
                    simdgroup_multiply_accumulate(
                        accumulators[row_fragment][column_fragment],
                        left[row_fragment],
                        right[column_fragment],
                        accumulators[row_fragment][column_fragment]);
                }
            }
        }
    }

    METAL_FUNC void store_result_safe(
        device OutT* output,
        int output_stride,
        short2 valid_shape) const {
        const int simdgroup_row = simdgroup_index / 2;
        const int simdgroup_column = simdgroup_index % 2;
        const ushort2 coordinate = course_matrix_coordinate(lane_index);

        #pragma unroll
        for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
            const int row =
                simdgroup_row * 16 + row_fragment * 8 + coordinate.y;
            if (row >= valid_shape.y) {
                continue;
            }
            #pragma unroll
            for (int column_fragment = 0; column_fragment < 2;
                 ++column_fragment) {
                const int column =
                    simdgroup_column * 16 + column_fragment * 8 + coordinate.x;
                if (column < valid_shape.x) {
                    output[row * output_stride + column] = OutT(
                        accumulators[row_fragment][column_fragment]
                            .thread_elements()[0]);
                }
                if (column + 1 < valid_shape.x) {
                    output[row * output_stride + column + 1] = OutT(
                        accumulators[row_fragment][column_fragment]
                            .thread_elements()[1]);
                }
            }
        }
    }
};

} // namespace tiny_llm
