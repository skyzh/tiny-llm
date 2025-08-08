from pathlib import Path
import shutil
from mlx import extension
from setuptools import Distribution
import inspect
import mlx
import os

if __name__ == "__main__":
    src_dir = Path(__file__).parent
    distribution = Distribution(
        {
            "name": "tiny_llm_ext",
            "ext_modules": [extension.CMakeExtension("tiny_llm_ext._ext")],
            "package_data": {"tiny_llm_ext": ["*.so", "*.dylib", "*.metallib"]},
        }
    )
    cmd = extension.CMakeBuild(distribution)
    cmd.initialize_options()
    cmd.build_temp = Path("build")
    cmd.build_lib = Path("build") / "lib"
    cmd.inplace = True
    cmd.ensure_finalized()
    cmd.run()
