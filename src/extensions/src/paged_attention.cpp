#include <stdexcept>
#include <string>

#include "tiny_llm_ext.h"

namespace tiny_llm_ext {

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

// Week 3, Day 3.
mx::array paged_cache_update(const mx::array &, const mx::array &, int, int, mx::StreamOrDevice) {
    checkpoint_todo("paged_cache_update", "Week 3, Day 3");
}

void PagedCacheUpdate::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("PagedCacheUpdate::eval_cpu", "Week 3, Day 3");
}

void PagedCacheUpdate::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("PagedCacheUpdate::eval_gpu", "Week 3, Day 3");
}

// Week 3, Day 4. Day 5 replaces the long-query schedule behind this API.
mx::array paged_attention(const mx::array &, const mx::array &, const mx::array &, const mx::array &, const mx::array &,
                          float, bool, int, int, mx::StreamOrDevice) {
    checkpoint_todo("paged_attention", "Week 3, Day 4");
}

void PagedAttention::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("PagedAttention::eval_cpu", "Week 3, Day 4");
}

void PagedAttention::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("PagedAttention::eval_gpu", "Week 3, Day 4");
}

}  // namespace tiny_llm_ext
