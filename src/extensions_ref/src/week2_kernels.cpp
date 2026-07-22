#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "mlx/backend/cpu/encoder.h"
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

template <typename T>
void rms_norm_cpu(const mx::array &x, const mx::array &weight, mx::array &out, float eps, mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(x);
    encoder.set_input_array(weight);
    encoder.set_output_array(out);
    encoder.dispatch([x = mx::array::unsafe_weak_copy(x), weight = mx::array::unsafe_weak_copy(weight),
                      out_ptr = out.data<T>(), eps]() {
        const T *x_ptr = x.data<T>();
        const T *weight_ptr = weight.data<T>();
        const int dim = x.shape().back();
        const int rows = x.size() / dim;
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < dim; ++col) {
                const float value = static_cast<float>(x_ptr[row * dim + col]);
                sum += value * value;
            }
            const float inv = 1.0f / std::sqrt(sum / dim + eps);
            for (int col = 0; col < dim; ++col) {
                const T normalized = static_cast<T>(static_cast<float>(x_ptr[row * dim + col]) * inv);
                out_ptr[row * dim + col] =
                    static_cast<T>(static_cast<float>(normalized) * static_cast<float>(weight_ptr[col]));
            }
        }
    });
}

template <typename T>
void rope_cpu(const mx::array &x, const mx::array &offsets, mx::array &out, int dims, float base, bool traditional,
              mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(x);
    encoder.set_input_array(offsets);
    encoder.set_output_array(out);
    encoder.dispatch([x = mx::array::unsafe_weak_copy(x), offsets = mx::array::unsafe_weak_copy(offsets),
                      out_ptr = out.data<T>(), dims, base, traditional]() {
        const T *x_ptr = x.data<T>();
        const int32_t *offsets_ptr = offsets.data<int32_t>();
        const int batch = x.shape()[0];
        const int length = x.shape()[1];
        const int heads = x.shape()[2];
        const int head_dim = x.shape()[3];
        const int half = dims / 2;
        for (int b = 0; b < batch; ++b) {
            for (int l = 0; l < length; ++l) {
                const float position = static_cast<float>(offsets_ptr[b] + l);
                for (int h = 0; h < heads; ++h) {
                    const int base_idx = ((b * length + l) * heads + h) * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        if (d >= dims) {
                            out_ptr[base_idx + d] = x_ptr[base_idx + d];
                            continue;
                        }
                        const int pair = traditional ? d / 2 : d % half;
                        const float angle = position * std::pow(base, -static_cast<float>(pair) / half);
                        const float c = std::cos(angle);
                        const float s = std::sin(angle);
                        int real_idx;
                        int imag_idx;
                        bool output_real;
                        if (traditional) {
                            real_idx = base_idx + pair * 2;
                            imag_idx = real_idx + 1;
                            output_real = (d % 2) == 0;
                        } else {
                            real_idx = base_idx + pair;
                            imag_idx = real_idx + half;
                            output_real = d < half;
                        }
                        const float real = static_cast<float>(x_ptr[real_idx]);
                        const float imag = static_cast<float>(x_ptr[imag_idx]);
                        out_ptr[base_idx + d] = static_cast<T>(output_real ? real * c - imag * s : imag * c + real * s);
                    }
                }
            }
        }
    });
}

template <typename T>
void swiglu_cpu(const mx::array &gate, const mx::array &up, mx::array &out, mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(gate);
    encoder.set_input_array(up);
    encoder.set_output_array(out);
    encoder.dispatch(
        [gate = mx::array::unsafe_weak_copy(gate), up = mx::array::unsafe_weak_copy(up), out_ptr = out.data<T>()]() {
            const T *gate_ptr = gate.data<T>();
            const T *up_ptr = up.data<T>();
            for (size_t i = 0; i < gate.size(); ++i) {
                const float g = static_cast<float>(gate_ptr[i]);
                out_ptr[i] = static_cast<T>((g / (1.0f + std::exp(-g))) * static_cast<float>(up_ptr[i]));
            }
        });
}

template <typename T>
void decode_attention_cpu(const mx::array &q, const mx::array &k, const mx::array &v, const mx::array &mask,
                          mx::array &out, float scale, bool is_causal, bool has_mask, int num_heads, int num_kv_heads,
                          mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_input_array(mask);
    encoder.set_output_array(out);
    encoder.dispatch([q = mx::array::unsafe_weak_copy(q), k = mx::array::unsafe_weak_copy(k),
                      v = mx::array::unsafe_weak_copy(v), mask = mx::array::unsafe_weak_copy(mask),
                      out_ptr = out.data<T>(), scale, is_causal, has_mask, num_heads, num_kv_heads]() {
        const T *q_ptr = q.data<T>();
        const T *k_ptr = k.data<T>();
        const T *v_ptr = v.data<T>();
        const float *mask_ptr = mask.data<float>();
        const int q_rows = q.shape()[0];
        const int length = q.shape()[1];
        const int context = k.shape()[1];
        const int dim = q.shape()[2];
        const int ratio = num_heads / num_kv_heads;
        for (int row = 0; row < q_rows; ++row) {
            const int batch = row / num_heads;
            const int q_head = row % num_heads;
            const int kv_row = batch * num_kv_heads + q_head / ratio;
            for (int l = 0; l < length; ++l) {
                float max_score = -std::numeric_limits<float>::infinity();
                float sum = 0.0f;
                std::vector<float> accum(dim, 0.0f);
                for (int s = 0; s < context; ++s) {
                    if (is_causal && s > context - length + l) {
                        continue;
                    }
                    float score = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        score += static_cast<float>(q_ptr[(row * length + l) * dim + d]) *
                                 static_cast<float>(k_ptr[(kv_row * context + s) * dim + d]);
                    }
                    score *= scale;
                    if (has_mask) {
                        score += mask_ptr[(row * length + l) * context + s];
                    }
                    const float new_max = std::max(max_score, score);
                    const float old_factor = std::exp(max_score - new_max);
                    const float score_factor = std::exp(score - new_max);
                    sum = sum * old_factor + score_factor;
                    for (int d = 0; d < dim; ++d) {
                        accum[d] = accum[d] * old_factor +
                                   score_factor * static_cast<float>(v_ptr[(kv_row * context + s) * dim + d]);
                    }
                    max_score = new_max;
                }
                for (int d = 0; d < dim; ++d) {
                    out_ptr[(row * length + l) * dim + d] = static_cast<T>(accum[d] / sum);
                }
            }
        }
    });
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
    if (outputs[0].dtype() == mx::float32) return rms_norm_cpu<float>(inputs[0], inputs[1], outputs[0], eps_, stream());
    if (outputs[0].dtype() == mx::float16)
        return rms_norm_cpu<mx::float16_t>(inputs[0], inputs[1], outputs[0], eps_, stream());
    return rms_norm_cpu<mx::bfloat16_t>(inputs[0], inputs[1], outputs[0], eps_, stream());
}

void Week2RoPE::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    if (outputs[0].dtype() == mx::float32)
        return rope_cpu<float>(inputs[0], inputs[1], outputs[0], dims_, base_, traditional_, stream());
    if (outputs[0].dtype() == mx::float16)
        return rope_cpu<mx::float16_t>(inputs[0], inputs[1], outputs[0], dims_, base_, traditional_, stream());
    return rope_cpu<mx::bfloat16_t>(inputs[0], inputs[1], outputs[0], dims_, base_, traditional_, stream());
}

void Week2SwiGLU::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    if (outputs[0].dtype() == mx::float32) return swiglu_cpu<float>(inputs[0], inputs[1], outputs[0], stream());
    if (outputs[0].dtype() == mx::float16) return swiglu_cpu<mx::float16_t>(inputs[0], inputs[1], outputs[0], stream());
    return swiglu_cpu<mx::bfloat16_t>(inputs[0], inputs[1], outputs[0], stream());
}

void Week2DecodeAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    if (outputs[0].dtype() == mx::float32)
        return decode_attention_cpu<float>(inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], scale_, is_causal_,
                                           has_mask_, num_heads_, num_kv_heads_, stream());
    if (outputs[0].dtype() == mx::float16)
        return decode_attention_cpu<mx::float16_t>(inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], scale_,
                                                   is_causal_, has_mask_, num_heads_, num_kv_heads_, stream());
    return decode_attention_cpu<mx::bfloat16_t>(inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], scale_,
                                                is_causal_, has_mask_, num_heads_, num_kv_heads_, stream());
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
    encoder.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(32, 1, 1));
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
    const size_t threads = std::min<size_t>(out.size(), kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(out.size(), 1, 1), MTL::Size(threads, 1, 1));
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
    encoder.dispatch_threadgroups(MTL::Size(q_rows * length, 1, 1), MTL::Size(128, 1, 1));
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
