#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

#include "mlx/ops.h"
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
    if (x.ndim() == 0 || x.shape().back() <= 0 || x.dtype() != weight.dtype() || weight.ndim() != 1 ||
        weight.shape()[0] != x.shape().back()) {
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

mx::array _qk_norm_rope4(const mx::array &normalized, const mx::array &offsets, int dims, float base, bool traditional,
                         mx::StreamOrDevice s) {
    require_float_dtype(normalized, "rope");
    if (normalized.ndim() != 4 || offsets.dtype() != mx::int32 || offsets.ndim() != 1 ||
        offsets.shape()[0] != normalized.shape()[0]) {
        throw std::runtime_error("rope: expected x=[B,L,H,D] and one int32 offset per batch row");
    }
    if (dims <= 0 || dims > normalized.shape()[3] || dims % 2 != 0) {
        throw std::runtime_error("rope: dims must be positive, even, and no larger than the head dimension");
    }
    auto target = to_stream(s);
    auto fallback = [&]() {
        return rope(mx::contiguous(normalized, false, target), offsets, dims, base, traditional, target);
    };
    if (normalized.dtype() != mx::bfloat16 || normalized.shape()[0] != 1 || normalized.shape()[1] != 1 ||
        (normalized.shape()[2] != 32 && normalized.shape()[2] != 8) || normalized.shape()[3] != 128 || dims != 128 ||
        traditional || !std::isfinite(base) || base <= 0.0f || target.device.type != mx::Device::gpu ||
        normalized.is_tracer() || normalized.status() != mx::array::Status::unscheduled ||
        !normalized.has_primitive() || !normalized.flags().row_contiguous || !offsets.flags().row_contiguous) {
        return fallback();
    }
    auto *norm = dynamic_cast<Week2RMSNorm *>(&normalized.primitive());
    if (!norm || !(norm->stream() == target) || normalized.inputs().size() != 2 ||
        !std::isfinite(norm->research_eps()) || norm->research_eps() <= 0.0f) {
        return fallback();
    }
    const auto &x = normalized.inputs()[0];
    const auto &weight = normalized.inputs()[1];
    if (x.shape() != normalized.shape() || x.dtype() != mx::bfloat16 || weight.dtype() != mx::bfloat16 ||
        weight.shape() != mx::Shape{128} || x.is_tracer() || weight.is_tracer() || !x.flags().row_contiguous ||
        !weight.flags().row_contiguous) {
        return fallback();
    }
    return mx::array(normalized.shape(), normalized.dtype(),
                     std::make_shared<Week2QKNormRoPE4>(target, norm->research_eps(), base), {x, weight, offsets});
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

mx::array _decode_attention_online(const mx::array &q, const mx::array &k, const mx::array &v,
                                    const mx::array &mask, float scale, bool is_causal, bool has_mask,
                                    int num_heads, int num_kv_heads, mx::StreamOrDevice s) {
    if (num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0 ||
        q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3 ||
        q.dtype() != mx::bfloat16 || k.dtype() != q.dtype() || v.dtype() != q.dtype() ||
        mask.dtype() != mx::float32 || q.shape()[1] != 1 || q.shape()[2] != 128 ||
        k.shape()[2] != 128 || k.shape() != v.shape() || k.shape()[1] <= 0 || q.shape()[0] <= 0 ||
        q.shape()[0] % num_heads != 0 || k.shape()[0] * num_heads != q.shape()[0] * num_kv_heads) {
        throw std::runtime_error("decode_attention_online: expected nonempty BF16 single-token D128 GQA");
    }
    if (!std::isfinite(scale) || !q.flags().row_contiguous || (has_mask && !mask.flags().row_contiguous)) {
        throw std::runtime_error("decode_attention_online: expected finite scale and contiguous query/mask");
    }
    if (has_mask && (mask.ndim() != 3 || mask.shape()[0] != q.shape()[0] ||
                     mask.shape()[1] != 1 || mask.shape()[2] != k.shape()[1])) {
        throw std::runtime_error("decode_attention_online: invalid mask shape");
    }
    return mx::array(q.shape(), q.dtype(),
        std::make_shared<Week2DecodeAttention>(to_stream(s), scale, is_causal, has_mask,
                                               num_heads, num_kv_heads, true), {q, k, v, mask});
}

mx::array _dense_attention_prefill_mma(const mx::array &q, const mx::array &k, const mx::array &v,
                                       const mx::array &mask, float scale, bool is_causal, bool has_mask, int num_heads,
                                       int num_kv_heads, mx::StreamOrDevice s) {
    if (q.dtype() != mx::bfloat16 || k.dtype() != mx::bfloat16 || v.dtype() != mx::bfloat16 ||
        mask.dtype() != mx::float32) {
        throw std::runtime_error("dense_attention_prefill_mma: expected BF16 q/k/v and float32 mask");
    }
    if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3 || q.shape()[2] != 128 || k.shape()[2] != 128 ||
        v.shape()[2] != 128 || k.shape() != v.shape() || q.shape()[1] <= 8 || num_heads % num_kv_heads != 0 ||
        q.shape()[0] % num_heads != 0 || k.shape()[0] * num_heads != q.shape()[0] * num_kv_heads) {
        throw std::runtime_error("dense_attention_prefill_mma: incompatible BF16/D128 attention shapes");
    }
    if (has_mask && (mask.ndim() != 3 || mask.shape()[0] != q.shape()[0] || mask.shape()[1] != q.shape()[1] ||
                     mask.shape()[2] != k.shape()[1])) {
        throw std::runtime_error("dense_attention_prefill_mma: mask must have shape [B*Hq,L,S]");
    }
    return mx::array(
        q.shape(), q.dtype(),
        std::make_shared<Week2DensePrefillMMA>(to_stream(s), scale, is_causal, has_mask, num_heads, num_kv_heads),
        {q, k, v, mask});
}

void Week2RMSNorm::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("rms_norm: the course extension is GPU-only");
}

void Week2RoPE::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("rope: the course extension is GPU-only");
}

void Week2QKNormRoPE4::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("qk_norm_rope4: the course extension is GPU-only");
}

void Week2SwiGLU::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("swiglu: the course extension is GPU-only");
}

void Week2DecodeAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("decode_attention: the course extension is GPU-only");
}

void Week2DensePrefillMMA::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("dense_attention_prefill_mma: the course extension is GPU-only");
}

#ifdef _METAL_

void Week2RMSNorm::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];
    const auto &weight = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    const int dim = x.shape().back();
    const bool register_cached = dim <= 4096;
    const auto kernel_name =
        std::string(register_cached ? "week2_rms_norm_register_cached_" : "week2_rms_norm_") + dtype_suffix(out);
    auto kernel = d.get_kernel(kernel_name, d.get_library("tiny_llm_ext_ref"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_output_array(out, 2);
    const int rows = x.size() / x.shape().back();
    encoder.set_bytes(rows, 3);
    encoder.set_bytes(dim, 4);
    encoder.set_bytes(eps_, 5);
    const int values_per_thread = register_cached ? 4 : 1;
    const int needed_threads = (dim + values_per_thread - 1) / values_per_thread;
    const int threads_per_threadgroup = register_cached ? ((needed_threads + 31) / 32) * 32 : 256;
    const int active_simdgroups = threads_per_threadgroup / 32;
    const size_t scratch_bytes = active_simdgroups * sizeof(float);
    const size_t aligned_scratch_bytes = (scratch_bytes + 15) / 16 * 16;
    encoder.set_threadgroup_memory_length(aligned_scratch_bytes, 0);
    encoder.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(threads_per_threadgroup, 1, 1));
}

void Week2RoPE::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];
    const auto &offsets = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(std::string("week2_rope_") + dtype_suffix(out), d.get_library("tiny_llm_ext_ref"));
    auto &encoder = mx::metal::get_command_encoder(stream());
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

void Week2QKNormRoPE4::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel("week2_qk_norm_rope4_bf16", d.get_library("tiny_llm_ext_ref"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(eps_, 4);
    encoder.set_bytes(base_, 5);
    const int dim = out.shape()[3];
    encoder.set_bytes(dim, 6);
    encoder.dispatch_threadgroups(MTL::Size(out.shape()[2] / 4, 1, 1), MTL::Size(128, 1, 1));
}

void Week2SwiGLU::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &gate = inputs[0];
    const auto &up = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(std::string("week2_swiglu_") + dtype_suffix(out), d.get_library("tiny_llm_ext_ref"));
    auto &encoder = mx::metal::get_command_encoder(stream());
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
    if (online_ && (!q.flags().row_contiguous || (has_mask_ && !mask.flags().row_contiguous))) {
        throw std::runtime_error("decode_attention_online: noncontiguous realized query/mask");
    }
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    const bool split = online_ && k.shape()[1] > 256 && k.shape()[1] <= 32768;
    const bool grouped_split = split && k.shape()[1] >= 8192 &&
        q.shape()[1] == 1 && q.shape()[2] == 128 && num_heads_ == 4 * num_kv_heads_;
    const int partitions = (k.shape()[1] + 255) / 256;
    std::optional<mx::array> partials;
    const std::string kernel_name = grouped_split ? "week2_decode_split_gqa4_bf16_d128" :
        split ? "week2_decode_split_bf16_d128" :
        online_ ? "week2_decode_online_bf16_d128" : std::string("week2_decode_attention_") + dtype_suffix(out);
    auto library = d.get_library("tiny_llm_ext_ref");
    MTL::ComputePipelineState* kernel;
    if (split) {
        const bool unit_dimension = k.strides()[2] == 1 && v.strides()[2] == 1;
        const bool sequence128 = k.strides()[1] == 128 && v.strides()[1] == 128;
        const bool causal = is_causal_;
        const bool masked = has_mask_;
        const mx::metal::MTLFCList constants = {
            {&unit_dimension, MTL::DataType::DataTypeBool, 100},
            {&causal, MTL::DataType::DataTypeBool, 101},
            {&masked, MTL::DataType::DataTypeBool, 102},
            {&sequence128, MTL::DataType::DataTypeBool, 103},
        };
        const std::string hash_name = kernel_name + (unit_dimension ? "_unit" : "_strided") +
            (causal ? "_causal" : "_noncausal") + (masked ? "_masked" : "_unmasked") +
            (sequence128 ? "_seq128" : "_seq_dynamic");
        kernel = d.get_kernel(kernel_name, library, hash_name, constants);
    } else {
        kernel = d.get_kernel(kernel_name, library);
    }
    auto &encoder = mx::metal::get_command_encoder(stream());
    if (split) {
        partials.emplace(mx::Shape{q.shape()[0], partitions, 130}, mx::float32, nullptr, std::vector<mx::array>{});
        partials->set_data(mx::allocator::malloc(partials->nbytes()));
        encoder.add_temporary(*partials);
    }
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_input_array(mask, 3);
    encoder.set_output_array(split ? *partials : out, 4);
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
    if (online_) {
        for (int axis = 0; axis < 3; ++axis) {
            encoder.set_bytes(k.strides()[axis], 14 + axis);
            encoder.set_bytes(v.strides()[axis], 17 + axis);
        }
    }
    if (split) encoder.set_bytes(partitions, 20);
    const int simdgroups_per_query = online_ ? 8 : 32;
    const int heads_per_group = grouped_split ? 4 : 1;
    encoder.set_threadgroup_memory_length(heads_per_group * simdgroups_per_query * (dim + 3) * sizeof(float), 0);
    encoder.dispatch_threadgroups(MTL::Size(q_rows * length / heads_per_group, split ? partitions : 1, 1),
                                  MTL::Size(simdgroups_per_query * 32, 1, 1));
    if (split) {
        auto merge = d.get_kernel("week2_decode_split_merge_bf16_d128", d.get_library("tiny_llm_ext_ref"));
        encoder.set_compute_pipeline_state(merge);
        encoder.set_input_array(*partials, 0);
        encoder.set_output_array(out, 1);
        encoder.set_bytes(partitions, 2);
        const size_t merge_scratch_bytes = (partitions + 2) * sizeof(float);
        const size_t aligned_merge_scratch_bytes = (merge_scratch_bytes + 15) / 16 * 16;
        encoder.set_threadgroup_memory_length(aligned_merge_scratch_bytes, 0);
        encoder.dispatch_threadgroups(MTL::Size(q_rows, 1, 1), MTL::Size(128, 1, 1));
    }
}

void Week2DensePrefillMMA::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &mask = inputs[3];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &d = mx::metal::device(stream().device);
    const int q_rows = q.shape()[0];
    const int length = q.shape()[1];
    const int context = k.shape()[1];
    constexpr int query_block = 32;
    constexpr int key_block = 16;
    const bool causal = is_causal_;
    const bool masked = has_mask_;
    const bool full_tiles = length % query_block == 0 && context % key_block == 0;
    const mx::metal::MTLFCList constants = {
        {&causal, MTL::DataType::DataTypeBool, 120},
        {&masked, MTL::DataType::DataTypeBool, 121},
        {&full_tiles, MTL::DataType::DataTypeBool, 122},
    };
    const std::string kernel_name = "week2_dense_prefill_mma_bf16_d128";
    const std::string hash_name = kernel_name + (causal ? "_causal" : "_noncausal") +
        (masked ? "_masked" : "_unmasked") + (full_tiles ? "_full" : "_tail");
    auto kernel = d.get_kernel(kernel_name, d.get_library("tiny_llm_ext_ref"), hash_name, constants);
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_input_array(mask, 3);
    encoder.set_output_array(out, 4);
    const int is_causal = is_causal_;
    const int has_mask = has_mask_;
    encoder.set_bytes(q_rows, 5);
    encoder.set_bytes(length, 6);
    encoder.set_bytes(context, 7);
    encoder.set_bytes(scale_, 8);
    encoder.set_bytes(is_causal, 9);
    encoder.set_bytes(has_mask, 10);
    encoder.set_bytes(num_heads_, 11);
    encoder.set_bytes(num_kv_heads_, 12);
    constexpr int threads = 128;
    encoder.dispatch_threadgroups(MTL::Size((length + query_block - 1) / query_block, q_rows, 1),
                                  MTL::Size(threads, 1, 1));
}

#else

void Week2RMSNorm::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}
void Week2RoPE::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}
void Week2QKNormRoPE4::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}
void Week2SwiGLU::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}
void Week2DecodeAttention::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}
void Week2DensePrefillMMA::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("Metal unavailable");
}
#endif

}  // namespace tiny_llm_ext_ref
