// Course-owned BM64 MLP QMM with the accepted shape and lifetime boundary.
#pragma once
#include <optional>

#include "tiny_llm_ext.h"
namespace tiny_llm_ext_ref {
std::optional<mx::array> _try_mlp_qmm_m64(const mx::array &, const mx::array &, const mx::array &, const mx::array &,
                                          int, int, mx::StreamOrDevice = {});
std::vector<int64_t> _mlp_qmm_m64_route_counts();
std::vector<int64_t> _mlp_qmm_m64_pipeline_state();
class MlpQmmM64 : public mx::Primitive {
    int K_, N_;

public:
    MlpQmmM64(mx::Stream s, int K, int N) : mx::Primitive(s), K_(K), N_(N) {}
    void eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) override;
    void eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) override;
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,
                                                             const std::vector<int> &) override {
        throw std::runtime_error("MlpQmmM64 has no vmap implementation.");
    }
    const char *name() const override { return "MlpQmmM64"; }
};
}  // namespace tiny_llm_ext_ref
