#pragma once

#include <optional>

#include "mlx/array.h"

namespace tiny_llm_ext_ref {

std::optional<mlx::core::array> try_week2_bf16_history_scores(
    const mlx::core::array &query, const mlx::core::array &key, const mlx::core::array &value,
    const std::optional<mlx::core::array> &mask);
mlx::core::array week2_bf16_history_values(const mlx::core::array &value, const mlx::core::array &weights);

}  // namespace tiny_llm_ext_ref
