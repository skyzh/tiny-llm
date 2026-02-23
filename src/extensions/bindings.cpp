// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "tiny_llm_ext.h"
#include "axpby.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "tiny-llm extensions for MLX";

    m.def("load_library", &tiny_llm_ext::load_library, "device"_a, "path"_a);

    m.def("axpby", &tiny_llm_ext::axpby, "x"_a, "y"_a, "alpha"_a, "beta"_a, nb::kw_only(), "stream"_a = nb::none(),
          R"(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            array: ``alpha * x + beta * y``
      )");

    m.def("quantized_matmul", &tiny_llm_ext::quantized_matmul, "scales"_a, "biases"_a, "group_size"_a, "bits"_a,
          "a"_a, "b"_a, "transpose_b"_a = false, "stream"_a = nb::none(),
          R"(
        Quantized matmul

        Args:
            scales (array): Scaling factors.
            biases (array): Biases.
            group_size (int): Group size.
            bits (int): Number of bits.
            a (array): Input array (activations).
            b (array): Input array (quantized weights).
            transpose_b (bool): Whether to transpose ``b``.

        Returns:
            array: Result of quantized matmul.
      )");

    m.def("flash_attention", &tiny_llm_ext::flash_attention, "query"_a, "key"_a, "value"_a, "mask"_a, "scale"_a = 1.0,
          "num_kv_heads"_a, "num_heads"_a, "stream"_a = nb::none(), R"(
        Flash attention layer

        Args:
            query (array): Query array.
            key (array): Key array.
            value (array): Value array.
            mask (array): Mask array.
            scale (float): Scaling factor.

        Returns:
            array: ``softmax(query @ key.T * scale) @ value``
      )");
}
