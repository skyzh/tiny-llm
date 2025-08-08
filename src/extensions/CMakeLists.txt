cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# ----------------------------- Extensions -----------------------------

# Add library
add_library(tiny_llm_ext)

# Add sources
target_sources(
  tiny_llm_ext
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/src/axpby.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp
)

# Add include headers
target_include_directories(tiny_llm_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/src)

# Link to mlx
target_link_libraries(tiny_llm_ext PUBLIC mlx)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # so that clangd can pick this up

# ----------------------------- Metal -----------------------------

# Build metallib
if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
    tiny_llm_ext_metallib
    TITLE
    tiny_llm_ext
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/axpby.metal
    INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(tiny_llm_ext tiny_llm_ext_metallib)
endif()

# ----------------------------- Python Bindings -----------------------------
nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE tiny_llm_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
