cmake_minimum_required(VERSION 4.1)
project(RenderingEngine CXX)

set(CMAKE_CXX_STANDARD 20)
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

# SIMD支持
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" LOWER_PROCESSOR)
if(LOWER_PROCESSOR MATCHES "x86_64|amd64|x64|i386|i686|x86")
    set(ENABLE_SIMD 1)  # x86
elseif (LOWER_PROCESSOR MATCHES "armv7|aarch64|arm64")
    set(ENABLE_SIMD 2)  # arm
else()
    set(ENABLE_SIMD 0)
endif()

# CUDA支持
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{FORCE_CPU})
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_FOUND)
        set(USE_CUDA 1)
        set(CMAKE_CUDA_ARCHITECTURES native)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # CUDA启用可重定位设备代码，device代码链接
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
    file(GLOB_RECURSE SRCcuda "${PROJECT_SOURCE_DIR}/Engine/CUDART/src/*.cu")
    target_sources(untitled PRIVATE ${SRCcuda})
    target_include_directories(untitled PRIVATE Engine/CUDART/include)
    target_link_libraries(untitled PRIVATE CUDA::cudart)  # 链接 CUDA 运行时
endif()

if(ENABLE_SIMD AND NOT MSVC)
    if (ENABLE_SIMD EQUAL 1)
        target_compile_options(untitled PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-msse4.1>
        )
    elseif (ENABLE_SIMD EQUAL 2)
        if(LOWER_PROCESSOR MATCHES "armv7")
            target_compile_options(untitled PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-mfpu=neon -mfloat-abi=hard>
            )
        else()  # aarch64 / arm64 / arm64e
            target_compile_options(untitled PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-march=armv8-a+simd>
            )
        endif()
    endif ()
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