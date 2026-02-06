cmake_minimum_required(VERSION 4.1)
project(RenderingEngine CXX)

set(CMAKE_CXX_STANDARD 20)
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

# SIMD支持
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64|i386|i686|x86")
    set(ENABLE_SSE ON)
else()
    set(ENABLE_SSE OFF)
endif()

# CUDA支持
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{FORCE_CPU})
    enable_language(CUDA)
    if(CUDAToolkit_FOUND AND NOT DEFINED ENV{FORCE_CPU})
    find_package(CUDAToolkit QUIET)
    set(USE_CUDA 1)
    else ()
        set(USE_CUDA 0)
    endif ()
else()
    set(USE_CUDA 0)
endif()

# 入口
file(GLOB_RECURSE SRC "${PROJECT_SOURCE_DIR}/*/src/*.cpp")
add_executable(untitled ${SRC})
target_compile_definitions(untitled PRIVATE USE_CUDA=${USE_CUDA})

if(USE_CUDA)
    file(GLOB_RECURSE SRCcuda "RenderEngine/Engine/CUDART/src/*.cu")
    target_sources(untitled PRIVATE ${SRCcuda})
    target_include_directories(untitled PUBLIC Engine/CUDART/include)
endif()

if(ENABLE_SSE AND NOT MSVC)
    target_compile_options(untitled PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-msse4.1>
    )
endif()

target_include_directories(untitled PUBLIC
        tool/include
        source/include
        math/include
        model/include
        Lib
        Engine/include
        shader/include
)