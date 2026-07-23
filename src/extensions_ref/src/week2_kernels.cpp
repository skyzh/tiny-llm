#include <algorithm>
#include <string>

#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext_ref {

namespace {

void require_float_dtype(const mx::array &x, const char *name) {
    if (x.dtype() != mx::float32 && x.dtype() != mx::float16 && x.dtype() != mx::bfloat16) {
        throw std::runtime_error(std::string(name) + ": expected float32, float16, or bfloat16");
    }
}

const char *dtype_suffix(const mx::array &x) {
    if (x.dtype() == mx::float32) {
        return "f32";
    }
    if (x.dtype() == mx::float16) {
        return "f16";
    }
    if (x.dtype() == mx::bfloat16) {
        return "bf16";
    }
    throw std::runtime_error("unsupported dtype");
}

}  // namespace

mx::array rms_norm(const mx::array &x, const mx::array &weight, float eps, mx::StreamOrDevice s) {
    require_float_dtype(x, "rms_norm");
    if (x.dtype() != weight.dtype() || weight.ndim() != 1 || weight.shape()[0] != x.shape().back()) {
        throw std::runtime_error("rms_norm: weight must match the input dtype and final dimension");
    }
    return mx::array(x.shape(), x.dtype(), std::make_shared<Week2RMSNorm>(to_stream(s), eps), {x, weight});
}

mx::array rope(const mx::array &x, const mx::array &offsets, int dims, float base, bool traditional,
               mx::StreamOrDevice s) {
    require_float_dtype(x, "rope");
    if (x.ndim() != 4 || offsets.dtype() != mx::int32 || offsets.ndim() != 1 || offsets.shape()[0] != x.shape()[0]) {
        throw std::runtime_error("rope: expected x=[B,L,H,D] and one int32 offset per batch row");
    }
    if (dims <= 0 || dims > x.shape()[3] || dims % 2 != 0) {
        throw std::runtime_error("rope: dims must be positive, even, and no larger than the head dimension");
    }
    return mx::array(x.shape(), x.dtype(), std::make_shared<Week2RoPE>(to_stream(s), dims, base, traditional),
                     {x, offsets});
}

mx::array swiglu(const mx::array &gate, const mx::array &up, mx::StreamOrDevice s) {
    require_float_dtype(gate, "swiglu");
    if (gate.dtype() != up.dtype() || gate.shape() != up.shape()) {
        throw std::runtime_error("swiglu: gate and up must have the same shape and dtype");
    }
    return mx::array(gate.shape(), gate.dtype(), std::make_shared<Week2SwiGLU>(to_stream(s)), {gate, up});
}

mx::array decode_attention(const mx::array &q, const mx::array &k, const mx::array &v, const mx::array &mask,
                           float scale, bool is_causal, bool has_mask, int num_heads, int num_kv_heads,
                           mx::StreamOrDevice s) {
    require_float_dtype(q, "decode_attention");
    if (q.dtype() != k.dtype() || q.dtype() != v.dtype() || mask.dtype() != mx::float32) {
        throw std::runtime_error("decode_attention: q, k, and v dtypes must match; mask must be float32");
    }
    if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3 || q.shape()[2] > 256 || q.shape()[2] != k.shape()[2] ||
        q.shape()[2] != v.shape()[2] || k.shape() != v.shape() || num_heads % num_kv_heads != 0) {
        throw std::runtime_error("decode_attention: incompatible attention shapes");
    }
    if (has_mask && (mask.ndim() != 3 || mask.shape()[0] != q.shape()[0] || mask.shape()[1] != q.shape()[1] ||
                     mask.shape()[2] != k.shape()[1])) {
        throw std::runtime_error("decode_attention: mask must have shape [B*Hq,L,S]");
    }
    return mx::array(
        q.shape(), q.dtype(),
        std::make_shared<Week2DecodeAttention>(to_stream(s), scale, is_causal, has_mask, num_heads, num_kv_heads),
        {q, k, v, mask});
}

void Week2RMSNorm::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("rms_norm: the course extension is GPU-only");
}

void Week2RoPE::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("rope: the course extension is GPU-only");
}

void Week2SwiGLU::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("swiglu: the course extension is GPU-only");
}

void Week2DecodeAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("decode_attention: the course extension is GPU-only");
}

#ifdef _METAL_

void Week2RMSNorm::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];
    const auto &weight = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(std::string("week2_rms_norm_") + dtype_suffix(out), d.get_library("tiny_llm_ext_ref"));
    auto &encoder = d.get_command_encoder(stream().index);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_output_array(out, 2);
    const int rows = x.size() / x.shape().back();
    const int dim = x.shape().back();
    encoder.set_bytes(rows, 3);
    encoder.set_bytes(dim, 4);
    encoder.set_bytes(eps_, 5);
    constexpr int threads_per_threadgroup = 256;
    constexpr int simdgroups_per_threadgroup = threads_per_threadgroup / 32;
    encoder.set_threadgroup_memory_length(simdgroups_per_threadgroup * sizeof(float), 0);
    encoder.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(threads_per_threadgroup, 1, 1));
}

void Week2RoPE::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];
    const auto &offsets = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(std::string("week2_rope_") + dtype_suffix(out), d.get_library("tiny_llm_ext_ref"));
    auto &encoder = d.get_command_encoder(stream().index);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(offsets, 1);
    encoder.set_output_array(out, 2);
    const int batch = x.shape()[0];
    const int length = x.shape()[1];
    const int heads = x.shape()[2];
    const int head_dim = x.shape()[3];
    const int traditional = traditional_;
    encoder.set_bytes(batch, 3);
    encoder.set_bytes(length, 4);
    encoder.set_bytes(heads, 5);
    encoder.set_bytes(head_dim, 6);
    encoder.set_bytes(dims_, 7);
    encoder.set_bytes(base_, 8);
    encoder.set_bytes(traditional, 9);
    constexpr int heads_per_thread = 4;
    const int head_blocks = (heads + heads_per_thread - 1) / heads_per_thread;
    const int work_items = batch * length * head_blocks * (dims_ / 2 + head_dim - dims_);
    const size_t threads = std::min<size_t>(work_items, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(work_items, 1, 1), MTL::Size(threads, 1, 1));
}

void Week2SwiGLU::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &gate = inputs[0];
    const auto &up = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(std::string("week2_swiglu_") + dtype_suffix(out), d.get_library("tiny_llm_ext_ref"));
    auto &encoder = d.get_command_encoder(stream().index);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(gate, 0);
    encoder.set_input_array(up, 1);
    encoder.set_output_array(out, 2);
    const int size = out.size();
    encoder.set_bytes(size, 3);
    const size_t threads = std::min<size_t>(out.size(), kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(out.size(), 1, 1), MTL::Size(threads, 1, 1));
}

void Week2DecodeAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &mask = inputs[3];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    auto kernel =
        d.get_kernel(std::string("week2_decode_attention_") + dtype_suffix(out), d.get_library("tiny_llm_ext_ref"));
    auto &encoder = d.get_command_encoder(stream().index);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_input_array(mask, 3);
    encoder.set_output_array(out, 4);
    const int q_rows = q.shape()[0];
    const int length = q.shape()[1];
    const int context = k.shape()[1];
    const int dim = q.shape()[2];
    const int is_causal = is_causal_;
    const int has_mask = has_mask_;
    encoder.set_bytes(q_rows, 5);
    encoder.set_bytes(length, 6);
    encoder.set_bytes(context, 7);
    encoder.set_bytes(dim, 8);
    encoder.set_bytes(num_heads_, 9);
    encoder.set_bytes(num_kv_heads_, 10);
    encoder.set_bytes(scale_, 11);
    encoder.set_bytes(is_causal, 12);
    encoder.set_bytes(has_mask, 13);
    constexpr int simdgroups_per_query = 32;
    encoder.set_threadgroup_memory_length(simdgroups_per_query * (dim + 3) * sizeof(float), 0);
    encoder.dispatch_threadgroups(MTL::Size(q_rows * length, 1, 1), MTL::Size(simdgroups_per_query * 32, 1, 1));
}

#else

void Week2RMSNorm::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}
void Week2RoPE::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}
void Week2SwiGLU::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}
void Week2DecodeAttention::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}

#endif

}  // namespace tiny_llm_ext_ref
