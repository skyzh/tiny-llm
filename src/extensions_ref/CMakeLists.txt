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
add_library(tiny_llm_ext_ref)

# Add sources
target_sources(
  tiny_llm_ext_ref
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/src/quantized_matmul.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/mlp_qmm_m64.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/week2_kernels.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/paged_attention.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp
)
target_sources(tiny_llm_ext_ref PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/src/week1_sampled_head.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/week2_bf16_history.cpp)

# Add include headers
target_include_directories(tiny_llm_ext_ref PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/src)

# Link to mlx
target_link_libraries(tiny_llm_ext_ref PUBLIC mlx)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # so that clangd can pick this up

# ----------------------------- Metal -----------------------------

# Build metallib
if(MLX_BUILD_METAL)
  set(WEEK1_HEAD_METAL "")
  set(WEEK2_HISTORY_METAL "")
  if(MLX_VERSION VERSION_EQUAL "0.32.4")
    list(GET MLX_INCLUDE_DIRS 0 INSTALLED_MLX_INCLUDE)
    set(WEEK1_HEAD_GENERATOR "${CMAKE_CURRENT_LIST_DIR}/generate_week1_sampled_head.py")
    set(WEEK1_HEAD_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/week1_sampled_head.metal")
    set(WEEK1_STEEL_DIR "${INSTALLED_MLX_INCLUDE}/mlx/backend/metal/kernels/steel/gemm")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      "${WEEK1_HEAD_GENERATOR}"
      "${WEEK1_STEEL_DIR}/kernels/steel_gemm_fused.h"
      "${WEEK1_STEEL_DIR}/mma.h"
      "${WEEK1_STEEL_DIR}/params.h")
    execute_process(
      COMMAND "${Python_EXECUTABLE}" "${WEEK1_HEAD_GENERATOR}" "${INSTALLED_MLX_INCLUDE}" "${WEEK1_HEAD_OUTPUT}"
      RESULT_VARIABLE WEEK1_HEAD_RESULT)
    if(WEEK1_HEAD_RESULT STREQUAL "0")
      set(WEEK1_HEAD_METAL "${WEEK1_HEAD_OUTPUT}")
      target_compile_definitions(tiny_llm_ext_ref PRIVATE TINY_LLM_WEEK1_SAMPLED_HEAD)
    elseif(NOT WEEK1_HEAD_RESULT STREQUAL "2")
      message(FATAL_ERROR "Could not derive the Week 1 sampled head")
    endif()

    set(WEEK2_HISTORY_GENERATOR "${CMAKE_CURRENT_LIST_DIR}/generate_week2_bf16_history.py")
    set(WEEK2_HISTORY_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/week2_bf16_history.metal")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      "${WEEK2_HISTORY_GENERATOR}"
      "${INSTALLED_MLX_INCLUDE}/mlx/backend/metal/kernels/gemv.h")
    execute_process(
      COMMAND "${Python_EXECUTABLE}" "${WEEK2_HISTORY_GENERATOR}" "${INSTALLED_MLX_INCLUDE}" "${WEEK2_HISTORY_OUTPUT}"
      RESULT_VARIABLE WEEK2_HISTORY_RESULT)
    if(WEEK2_HISTORY_RESULT STREQUAL "0")
      set(WEEK2_HISTORY_METAL "${WEEK2_HISTORY_OUTPUT}")
      target_compile_definitions(tiny_llm_ext_ref PRIVATE TINY_LLM_WEEK2_BF16_HISTORY)
    elseif(NOT WEEK2_HISTORY_RESULT STREQUAL "2")
      message(FATAL_ERROR "Could not derive the Week 2 BF16 history loads")
    endif()
  endif()
  mlx_build_metallib(
    TARGET
    tiny_llm_ext_ref_metallib
    TITLE
    tiny_llm_ext_ref
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/quantized_matmul.metal
    ${CMAKE_CURRENT_LIST_DIR}/src/week2_kernels.metal
    ${CMAKE_CURRENT_LIST_DIR}/src/paged_attention.metal
    ${WEEK1_HEAD_METAL}
    ${WEEK2_HISTORY_METAL}
    INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(tiny_llm_ext_ref tiny_llm_ext_ref_metallib)
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
target_link_libraries(_ext PRIVATE tiny_llm_ext_ref)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
