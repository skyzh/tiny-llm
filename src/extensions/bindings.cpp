// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "axpby.h"
#include "tiny_llm_ext.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "tiny-llm extensions for MLX";

    m.def("load_library", &tiny_llm_ext::load_library, "path"_a);

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

    // Week 2, Day 3. Days 6-7 extend the schedule behind this stable binding.
    m.def("quantized_matmul", &tiny_llm_ext::quantized_matmul, "scales"_a, "biases"_a, "group_size"_a, "bits"_a, "a"_a,
          "b"_a, "transpose_b"_a = false, "use_simdgroup"_a = true, "use_split_k"_a = false, "stream"_a = nb::none());

    // Week 3, Day 4. Earlier checkpoints keep the readable selected-row lookup.
    m.def("quantized_embedding", &tiny_llm_ext::quantized_embedding, "indices"_a, "scales"_a, "biases"_a, "weight"_a,
          "group_size"_a, "bits"_a, "stream"_a = nb::none());

    // Week 2, Day 4.
    m.def("rms_norm", &tiny_llm_ext::rms_norm, "x"_a, "weight"_a, "eps"_a, "stream"_a = nb::none());
    m.def("rope", &tiny_llm_ext::rope, "x"_a, "offsets"_a, "dims"_a, "base"_a, "traditional"_a = false,
          "stream"_a = nb::none());
    m.def("swiglu", &tiny_llm_ext::swiglu, "gate"_a, "up"_a, "stream"_a = nb::none());

    // Week 2, Day 5.
    m.def("decode_attention", &tiny_llm_ext::decode_attention, "query"_a, "key"_a, "value"_a, "mask"_a, "scale"_a,
          "is_causal"_a, "has_mask"_a, "num_heads"_a, "num_kv_heads"_a, "stream"_a = nb::none());

    // Week 3, Day 3.
    m.def("paged_cache_update", &tiny_llm_ext::paged_cache_update, "pages"_a, "values"_a, "page_id"_a, "start"_a,
          "stream"_a = nb::none());

    // Week 3, Day 4. Day 5 replaces only the long-query schedule.
    m.def("paged_attention", &tiny_llm_ext::paged_attention, "query"_a, "key_pages"_a, "value_pages"_a, "block_table"_a,
          "context_lens"_a, "scale"_a = 1.0, "is_causal"_a = false, "num_kv_heads"_a, "num_heads"_a,
          "stream"_a = nb::none());
}
