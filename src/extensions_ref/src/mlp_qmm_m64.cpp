#include "mlp_qmm_m64.h"

#include <array>
#include <atomic>
#include <mutex>
#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif
namespace tiny_llm_ext_ref {
namespace {
std::array<std::atomic<int64_t>, 6> routes{};
std::mutex pipeline_mutex;
std::array<int64_t, 5> pipeline_state{{0, 0, 0, 7808, 0}};
bool owned_data(const mx::array &a) {
    return !a.is_tracer() && a.data_shared_ptr() && a.data_shared_ptr()->is_owned() && a.buffer().ptr() != nullptr &&
           a.offset() >= 0 && static_cast<__int128>(a.offset()) + a.nbytes() <= a.buffer_size();
}
bool captured(const mx::array &a, const mx::Stream &target, int K) {
    if (a.has_primitive() && !(a.primitive().stream() == target)) return false;
    if (owned_data(a)) return true;
    if (a.is_tracer() || a.status() != mx::array::Status::unscheduled || !a.has_primitive()) return false;
    return K == 2560 ? dynamic_cast<Week2RMSNorm *>(&a.primitive()) != nullptr
                     : dynamic_cast<Week2SwiGLU *>(&a.primitive()) != nullptr;
}
}  // namespace
std::vector<int64_t> _mlp_qmm_m64_route_counts() {
    std::vector<int64_t> out;
    for (auto &r : routes) out.push_back(r.load(std::memory_order_relaxed));
    return out;
}
std::vector<int64_t> _mlp_qmm_m64_pipeline_state() {
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    return std::vector<int64_t>(pipeline_state.begin(), pipeline_state.end());
}
std::optional<mx::array> _try_mlp_qmm_m64(const mx::array &a, const mx::array &w, const mx::array &scales,
                                          const mx::array &biases, int group_size, int bits, mx::StreamOrDevice sd) {
    routes[0].fetch_add(1, std::memory_order_relaxed);
    auto fallback = []() -> std::optional<mx::array> {
        routes[3].fetch_add(1, std::memory_order_relaxed);
        return std::nullopt;
    };
    auto target = to_stream(sd);
    if (target.device.type != mx::Device::gpu || group_size != 128 || bits != 4 || a.ndim() != 3 || a.shape(0) != 1 ||
        a.shape(1) != 256 || w.ndim() != 2)
        return fallback();
    const int K = a.shape(2), N = w.shape(0);
    if (!((K == 2560 && N == 9728) || (K == 9728 && N == 2560)) || w.shape() != mx::Shape{N, K / 8} ||
        scales.shape() != mx::Shape{N, K / 128} || biases.shape() != scales.shape())
        return fallback();
    if (a.dtype() != mx::bfloat16 || w.dtype() != mx::uint32 || scales.dtype() != mx::bfloat16 ||
        biases.dtype() != mx::bfloat16)
        return fallback();
    for (const auto *v : {&a, &w, &scales, &biases})
        if (v->is_tracer() || !v->flags().row_contiguous) return fallback();
    if (!owned_data(w) || !owned_data(scales) || !owned_data(biases) || !captured(a, target, K)) return fallback();
    const mx::array *values[] = {&a, &w, &scales, &biases};
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            if (owned_data(*values[i]) && owned_data(*values[j]) &&
                values[i]->data_shared_ptr() == values[j]->data_shared_ptr())
                return fallback();
#ifdef _METAL_
    int gen = mx::metal::device(target.device).get_architecture_gen();
    if (gen <= 0 || gen >= 17) return fallback();
#else
    return fallback();
#endif
    routes[1].fetch_add(1, std::memory_order_relaxed);
    return mx::array({1, 256, N}, a.dtype(), std::make_shared<MlpQmmM64>(target, K, N), {a, w, scales, biases});
}
void MlpQmmM64::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("MlpQmmM64 is GPU-only");
}
void MlpQmmM64::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
#ifdef _METAL_
    const auto &a = inputs[0];
    const auto &w = inputs[1];
    const auto &scales = inputs[2];
    const auto &biases = inputs[3];
    auto &out = outputs[0];
    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    if (d.get_architecture_gen() <= 0 || d.get_architecture_gen() >= 17)
        throw std::runtime_error("MlpQmmM64 selector identity changed");
    for (const auto &v : inputs) {
        if (!owned_data(v) || !v.flags().row_contiguous) throw std::runtime_error("MlpQmmM64 owner/layout changed");
        auto *b = static_cast<const MTL::Buffer *>(v.buffer().ptr());
        if (!b || v.offset() > b->length() || v.nbytes() > b->length() - v.offset())
            throw std::runtime_error("MlpQmmM64 view bounds");
    }
    auto library = d.get_library("tiny_llm_ext_ref");
    auto kernel = d.get_kernel("mlp_qmm_m64_bf16", library);
    const int64_t max_threads = kernel->maxTotalThreadsPerThreadgroup(),
                  actual_shared = kernel->staticThreadgroupMemoryLength(),
                  device_max = d.mtl_device()->maxThreadgroupMemoryLength();
    const bool resource_ok = max_threads >= 128 && 7808 <= device_max && actual_shared <= device_max;
    {
        std::lock_guard<std::mutex> lock(pipeline_mutex);
        pipeline_state[0] = max_threads;
        pipeline_state[1] = actual_shared;
        pipeline_state[2] = device_max;
        if (!resource_ok) pipeline_state[4]++;
    }
    if (!resource_ok) throw std::runtime_error("MlpQmmM64 compiled pipeline resource failure");
    out.set_data(mx::allocator::malloc(out.nbytes()));
    for (const auto &v : inputs)
        if (out.buffer().ptr() == v.buffer().ptr()) throw std::runtime_error("MlpQmmM64 output aliases input");
    auto &e = mx::metal::get_command_encoder(s);
    e.set_compute_pipeline_state(kernel);
    e.set_input_array(w, 0);
    e.set_input_array(scales, 1);
    e.set_input_array(biases, 2);
    e.set_input_array(a, 3);
    e.set_output_array(out, 4);
    e.set_bytes(K_, 5);
    e.set_bytes(N_, 6);
    e.set_bytes(256, 7);
    e.dispatch_threadgroups(MTL::Size(N_ / 32, 4, 1), MTL::Size(32, 2, 2));
    routes[2].fetch_add(1, std::memory_order_relaxed);
    routes[N_ == 9728 ? 4 : 5].fetch_add(1, std::memory_order_relaxed);
#else
    throw std::runtime_error("MlpQmmM64 Metal backend required");
#endif
}
}  // namespace tiny_llm_ext_ref
