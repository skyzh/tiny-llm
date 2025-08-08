#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext_ref {

void load_library(mx::Device d, const char *path);

mx::array quantized_matmul(const mx::array &scales,   // Input array scales
                           const mx::array &biases,   // Input array biases
                           const int group_size,      // Group size
                           const int bits,            // Number of bits
                           const mx::array &a,        // Input array a (not quantized)
                           const mx::array &b,        // Input array b (quantized)
                           const bool transpose_b,    // Whether to transpose b
                           mx::StreamOrDevice s = {}  // Stream on which to schedule the operation
);

class QuantizedMatmul : public mx::Primitive {
public:
    explicit QuantizedMatmul(mx::Stream stream, const int group_size, const int bits)
        : mx::Primitive(stream), group_size_(group_size), bits_(bits) {};

    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                             const std::vector<int> &axes) override {
        throw std::runtime_error("QuantizedMatmul has no vmap implementation.");
    }

    const char *name() const override { return "QuantizedMatmul"; }

private:
    int group_size_;
    int bits_;
};

mx::array flash_attention(const mx::array &q, const mx::array &k, const mx::array &v, const mx::array &mask,
                          const float scale, const int num_kv_heads, const int num_heads, mx::StreamOrDevice s = {});

mx::array flash_attention_no_mask(const mx::array &q, const mx::array &k, const mx::array &v,
                                  const float scale, const int num_kv_heads, const int num_heads, mx::StreamOrDevice s = {});

class FlashAttention : public mx::Primitive {
public:
    explicit FlashAttention(mx::Stream stream, const float scale, const int num_kv_heads, const int num_heads)
        : mx::Primitive(stream), scale_(scale), num_kv_heads_(num_kv_heads), num_heads_(num_heads) {};

    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                             const std::vector<int> &axes) override {
        throw std::runtime_error("FlashAttention has no vmap implementation.");
    }

    const char *name() const override { return "FlashAttention"; }

private:
    float scale_;
    int num_kv_heads_;
    int num_heads_;
};

}  // namespace tiny_llm_ext_ref
