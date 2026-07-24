#pragma once

#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace tiny_llm_ext_ref {

void load_library(const char *path);

mx::array quantized_matmul(const mx::array &scales,  // Input array scales
                           const mx::array &biases,  // Input array biases
                           const int group_size,     // Group size
                           const int bits,           // Number of bits
                           const mx::array &a,       // Input array a (not quantized)
                           const mx::array &b,       // Input array b (quantized)
                           const bool transpose_b,   // Whether to transpose b
                           const bool use_simdgroup = true, const bool use_split_k = false,
                           mx::StreamOrDevice s = {}  // Stream on which to schedule the operation
);

class QuantizedMatmul : public mx::Primitive {
public:
    QuantizedMatmul(mx::Stream stream, bool use_simdgroup, bool use_split_k)
        : mx::Primitive(stream), use_simdgroup_(use_simdgroup), use_split_k_(use_split_k) {};

    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                             const std::vector<int> &axes) override {
        throw std::runtime_error("QuantizedMatmul has no vmap implementation.");
    }

    const char *name() const override { return "QuantizedMatmul"; }

private:
    bool use_simdgroup_;
    bool use_split_k_;
};

mx::array quantized_embedding(const mx::array &indices, const mx::array &scales, const mx::array &biases,
                              const mx::array &weight, int group_size, int bits, mx::StreamOrDevice s = {});

class QuantizedEmbedding : public mx::Primitive {
public:
    explicit QuantizedEmbedding(mx::Stream stream) : mx::Primitive(stream) {}
    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,
                                                             const std::vector<int> &) override {
        throw std::runtime_error("QuantizedEmbedding has no vmap implementation.");
    }
    const char *name() const override { return "QuantizedEmbedding"; }
};

mx::array rms_norm(const mx::array &x, const mx::array &weight, float eps, mx::StreamOrDevice s = {});
mx::array rope(const mx::array &x, const mx::array &offsets, int dims, float base, bool traditional,
               mx::StreamOrDevice s = {});
mx::array swiglu(const mx::array &gate, const mx::array &up, mx::StreamOrDevice s = {});
mx::array decode_attention(const mx::array &q, const mx::array &k, const mx::array &v, const mx::array &mask,
                           float scale, bool is_causal, bool has_mask, int num_heads, int num_kv_heads,
                           mx::StreamOrDevice s = {});

class Week2RMSNorm : public mx::Primitive {
public:
    Week2RMSNorm(mx::Stream stream, float eps) : mx::Primitive(stream), eps_(eps) {}
    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,
                                                             const std::vector<int> &) override {
        throw std::runtime_error("Week2RMSNorm has no vmap implementation.");
    }
    const char *name() const override { return "Week2RMSNorm"; }

private:
    float eps_;
};

class Week2RoPE : public mx::Primitive {
public:
    Week2RoPE(mx::Stream stream, int dims, float base, bool traditional)
        : mx::Primitive(stream), dims_(dims), base_(base), traditional_(traditional) {}
    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,
                                                             const std::vector<int> &) override {
        throw std::runtime_error("Week2RoPE has no vmap implementation.");
    }
    const char *name() const override { return "Week2RoPE"; }

private:
    int dims_;
    float base_;
    bool traditional_;
};

class Week2SwiGLU : public mx::Primitive {
public:
    explicit Week2SwiGLU(mx::Stream stream) : mx::Primitive(stream) {}
    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,
                                                             const std::vector<int> &) override {
        throw std::runtime_error("Week2SwiGLU has no vmap implementation.");
    }
    const char *name() const override { return "Week2SwiGLU"; }
};

class Week2DecodeAttention : public mx::Primitive {
public:
    Week2DecodeAttention(mx::Stream stream, float scale, bool is_causal, bool has_mask, int num_heads, int num_kv_heads)
        : mx::Primitive(stream),
          scale_(scale),
          is_causal_(is_causal),
          has_mask_(has_mask),
          num_heads_(num_heads),
          num_kv_heads_(num_kv_heads) {}
    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,
                                                             const std::vector<int> &) override {
        throw std::runtime_error("Week2DecodeAttention has no vmap implementation.");
    }
    const char *name() const override { return "Week2DecodeAttention"; }

private:
    float scale_;
    bool is_causal_;
    bool has_mask_;
    int num_heads_;
    int num_kv_heads_;
};

mx::array paged_attention(const mx::array &q, const mx::array &key_pages, const mx::array &value_pages,
                          const mx::array &block_table, const mx::array &context_lens, const float scale,
                          const bool is_causal, const int num_kv_heads, const int num_heads, mx::StreamOrDevice s = {});

mx::array paged_cache_update(const mx::array &pages, const mx::array &values, int page_id, int start,
                             mx::StreamOrDevice s = {});

class PagedCacheUpdate : public mx::Primitive {
public:
    PagedCacheUpdate(mx::Stream stream, int page_id, int start)
        : mx::Primitive(stream), page_id_(page_id), start_(start) {}
    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,
                                                             const std::vector<int> &) override {
        throw std::runtime_error("PagedCacheUpdate has no vmap implementation.");
    }
    const char *name() const override { return "PagedCacheUpdate"; }

private:
    int page_id_;
    int start_;
};

class PagedAttention : public mx::Primitive {
public:
    explicit PagedAttention(mx::Stream stream, const float scale, const bool is_causal, const int num_kv_heads,
                            const int num_heads)
        : mx::Primitive(stream),
          scale_(scale),
          is_causal_(is_causal),
          num_kv_heads_(num_kv_heads),
          num_heads_(num_heads) {};

    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;

    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                             const std::vector<int> &axes) override {
        throw std::runtime_error("PagedAttention has no vmap implementation.");
    }

    const char *name() const override { return "PagedAttention"; }

private:
    float scale_;
    bool is_causal_;
    int num_kv_heads_;
    int num_heads_;
};

}  // namespace tiny_llm_ext_ref
