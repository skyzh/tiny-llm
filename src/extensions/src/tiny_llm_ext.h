#pragma once

#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

void load_library(const char *path);

// Week 2, Day 3: implement the wrapper and the initial vanilla/matvec paths.
// Week 2, Days 6-7: extend the same interface with SIMD-matrix and Split-K scheduling.
mx::array quantized_matmul(const mx::array &scales, const mx::array &biases, const int group_size, const int bits,
                           const mx::array &a, const mx::array &b, const bool transpose_b,
                           const bool use_simdgroup = true, const bool use_split_k = false, mx::StreamOrDevice s = {});

class QuantizedMatmul : public mx::Primitive {
public:
    QuantizedMatmul(mx::Stream stream, bool use_simdgroup, bool use_split_k)
        : mx::Primitive(stream), use_simdgroup_(use_simdgroup), use_split_k_(use_split_k) {}

    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,
                                                             const std::vector<int> &) override {
        throw std::runtime_error("QuantizedMatmul has no vmap implementation.");
    }
    const char *name() const override { return "QuantizedMatmul"; }

private:
    bool use_simdgroup_;
    bool use_split_k_;
};

// Week 3, Day 4: replace the readable selected-row lookup with one extension dispatch.
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

// Week 2, Day 4: implement the three fused model kernels in checkpoint order.
mx::array rms_norm(const mx::array &x, const mx::array &weight, float eps, mx::StreamOrDevice s = {});
mx::array rope(const mx::array &x, const mx::array &offsets, int dims, float base, bool traditional,
               mx::StreamOrDevice s = {});
mx::array swiglu(const mx::array &gate, const mx::array &up, mx::StreamOrDevice s = {});

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

// Week 2, Day 5: implement online-softmax decode attention.
mx::array decode_attention(const mx::array &q, const mx::array &k, const mx::array &v, const mx::array &mask,
                           float scale, bool is_causal, bool has_mask, int num_heads, int num_kv_heads,
                           mx::StreamOrDevice s = {});

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

// Week 3, Day 3: implement slice-sized writes into paged KV storage.
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

// Week 3, Day 4: implement direct paged decode and correctness-first prefill.
// Week 3, Day 5: optimize the long-query prefill schedule without changing this API.
mx::array paged_attention(const mx::array &q, const mx::array &key_pages, const mx::array &value_pages,
                          const mx::array &block_table, const mx::array &context_lens, const float scale,
                          const bool is_causal, const int num_kv_heads, const int num_heads, mx::StreamOrDevice s = {});

class PagedAttention : public mx::Primitive {
public:
    PagedAttention(mx::Stream stream, float scale, bool is_causal, int num_kv_heads, int num_heads)
        : mx::Primitive(stream),
          scale_(scale),
          is_causal_(is_causal),
          num_kv_heads_(num_kv_heads),
          num_heads_(num_heads) {}
    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,
                                                             const std::vector<int> &) override {
        throw std::runtime_error("PagedAttention has no vmap implementation.");
    }
    const char *name() const override { return "PagedAttention"; }

private:
    float scale_;
    bool is_causal_;
    int num_kv_heads_;
    int num_heads_;
};

}  // namespace tiny_llm_ext
