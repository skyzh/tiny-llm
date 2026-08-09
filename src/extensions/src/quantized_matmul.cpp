#include <stdexcept>
#include <string>

#include "tiny_llm_ext.h"

namespace tiny_llm_ext {

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

// Week 2, Day 2. Days 5 and 6 extend the dispatch policy behind this API.
mx::array quantized_matmul(const mx::array &, const mx::array &, int, int, const mx::array &, const mx::array &, bool,
                           bool, bool, mx::StreamOrDevice) {
    checkpoint_todo("quantized_matmul", "Week 2, Day 2");
}

void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedMatmul::eval_cpu", "Week 2, Day 2");
}

void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedMatmul::eval_gpu", "Week 2, Day 2");
}

// Week 3, Day 4. The earlier Week 2 checkpoints keep the readable row lookup.
mx::array quantized_embedding(const mx::array &, const mx::array &, const mx::array &, const mx::array &, int, int,
                              mx::StreamOrDevice) {
    checkpoint_todo("quantized_embedding", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_cpu", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_gpu", "Week 3, Day 4");
}

}  // namespace tiny_llm_ext
