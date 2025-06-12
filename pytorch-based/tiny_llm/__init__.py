from .attention import *
from .basics import *
from .embedding import *
from .layer_norm import *
from .positional_encoding import *

try:
    from .quantize import *
except ImportError:
    pass

from .qwen2_week1 import *

try:
    from .generate import *
except ImportError:
    def simple_generate(*args, **kwargs):
        return "Generate function not available - missing dependencies"
    
    def simple_generate_with_kv_cache(*args, **kwargs):
        return "KV cache generation not available - missing dependencies"


try:
    from .qwen2_week2 import Qwen2ModelWeek2
except ImportError:
    class Qwen2ModelWeek2:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Qwen2ModelWeek2 not available - missing MLX dependencies")
        
        def __call__(self, *args, **kwargs):
            raise NotImplementedError("Qwen2ModelWeek2 not available - missing MLX dependencies")
