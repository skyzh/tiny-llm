#include <stdexcept>
#include <string>

#include "tiny_llm_ext.h"

namespace tiny_llm_ext {

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

// Week 2, Day 4.
mx::array rms_norm(const mx::array &, const mx::array &, float, mx::StreamOrDevice) {
    checkpoint_todo("rms_norm", "Week 2, Day 4");
}

mx::array rope(const mx::array &, const mx::array &, int, float, bool, mx::StreamOrDevice) {
    checkpoint_todo("rope", "Week 2, Day 4");
}

mx::array swiglu(const mx::array &, const mx::array &, mx::StreamOrDevice) {
    checkpoint_todo("swiglu", "Week 2, Day 4");
}

void Week2RMSNorm::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2RMSNorm::eval_cpu", "Week 2, Day 4");
}

void Week2RMSNorm::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2RMSNorm::eval_gpu", "Week 2, Day 4");
}

void Week2RoPE::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2RoPE::eval_cpu", "Week 2, Day 4");
}

void Week2RoPE::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2RoPE::eval_gpu", "Week 2, Day 4");
}

void Week2SwiGLU::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2SwiGLU::eval_cpu", "Week 2, Day 4");
}

void Week2SwiGLU::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2SwiGLU::eval_gpu", "Week 2, Day 4");
}

// Week 2, Day 5.
mx::array decode_attention(const mx::array &, const mx::array &, const mx::array &, const mx::array &, float, bool,
                           bool, int, int, mx::StreamOrDevice) {
    checkpoint_todo("decode_attention", "Week 2, Day 5");
}

void Week2DecodeAttention::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2DecodeAttention::eval_cpu", "Week 2, Day 5");
}

void Week2DecodeAttention::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("Week2DecodeAttention::eval_gpu", "Week 2, Day 5");
}

}  // namespace tiny_llm_ext
