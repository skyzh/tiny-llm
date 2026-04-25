// Copyright © 2023-2025 Apple Inc.

#include <stdexcept>

#include "tiny_llm_ext.h"

namespace tiny_llm_ext {

mx::array flash_attention(const mx::array &q, const mx::array &k, const mx::array &v, const mx::array &mask,
                          const float scale, const bool is_causal, const int num_kv_heads, const int num_heads,
                          mx::StreamOrDevice s /* = {} */) {
    // TODO(student): implement flash attention.
    (void)q;
    (void)k;
    (void)v;
    (void)mask;
    (void)scale;
    (void)is_causal;
    (void)num_kv_heads;
    (void)num_heads;
    (void)s;
    throw std::runtime_error("flash_attention is not implemented.");
}

void FlashAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    // TODO(student): implement CPU kernel.
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("FlashAttention::eval_cpu is not implemented.");
}

#ifdef _METAL_

void FlashAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    // TODO(student): implement Metal kernel dispatch.
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("FlashAttention::eval_gpu is not implemented.");
}

#else

void FlashAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    (void)inputs;
    (void)outputs;
    throw std::runtime_error("FlashAttention has no GPU implementation.");
}

#endif

}  // namespace tiny_llm_ext
