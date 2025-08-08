#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext_ref {

void load_library(mx::Device d, const char *path) {
#ifdef _METAL_
    auto &md = mx::metal::device(d);
    md.get_library("tiny_llm_ext_ref", path);
#endif
}

}  // namespace tiny_llm_ext_ref
