#pragma once

#include <stdexcept>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

void load_library(mx::Device d, const char *path);

///////////////////////////////////////////////////////////////////////////////
// Flash Attention (student implementation)
///////////////////////////////////////////////////////////////////////////////

mx::array flash_attention(const mx::array &q, const mx::array &k, const mx::array &v, const mx::array &mask,
                          const float scale, const bool is_causal, const int num_kv_heads, const int num_heads,
                          mx::StreamOrDevice s = {});

class FlashAttention : public mx::Primitive {
public:
    explicit FlashAttention(mx::Stream stream, const float scale, const bool is_causal, const int num_kv_heads,
                            const int num_heads)
        : mx::Primitive(stream), scale_(scale), is_causal_(is_causal), num_kv_heads_(num_kv_heads), num_heads_(num_heads) {};

    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                             const std::vector<int> &axes) override {
        throw std::runtime_error("FlashAttention has no vmap implementation.");
    }

    const char *name() const override { return "FlashAttention"; }

private:
    float scale_;
    bool is_causal_;
    int num_kv_heads_;
    int num_heads_;
};

}  // namespace tiny_llm_ext
