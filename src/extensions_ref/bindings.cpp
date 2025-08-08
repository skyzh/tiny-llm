// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "tiny_llm_ext.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "tiny-llm extensions for MLX";

    m.def("load_library", &tiny_llm_ext_ref::load_library, "device"_a, "path"_a);

    m.def("quantized_matmul", &tiny_llm_ext_ref::quantized_matmul, "scales"_a, "biases"_a, "group_size"_a, "bits"_a,
          "a"_a, "b"_a, "transpose_b"_a = false, "stream"_a = nb::none(),
          R"(
        Quantized matmul layer

        Args:
            scales (array): Scaling factors for ``a``.
            biases (array): Biases for ``a``.
            group_size (int): Group size for ``a``.
            bits (int): Number of bits for ``a``.
            a (array): Input array.
            b (array): Input array.
            transpose_b (bool): Whether to transpose ``b`` before multiplication.

        Returns:
            array: ``a * b``
      )");

    m.def("flash_attention", &tiny_llm_ext_ref::flash_attention, "query"_a, "key"_a, "value"_a, "mask"_a, "scale"_a = 1.0,
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
