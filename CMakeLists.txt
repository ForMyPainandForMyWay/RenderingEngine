cmake_minimum_required(VERSION 4.1)
project(RenderingEngine)

set(CMAKE_CXX_STANDARD 20)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|x64")
    set(ENABLE_SSE ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i386|i686|x86")
    set(ENABLE_SSE ON)
else()
    set(ENABLE_SSE OFF)
endif()


set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

file(GLOB_RECURSE SRC "${PROJECT_SOURCE_DIR}/*/src/*.cpp")


add_executable(untitled ${SRC})

if(ENABLE_SSE AND NOT MSVC)
    target_compile_options(untitled PRIVATE -msse4.1)
endif()

target_include_directories(untitled PUBLIC
        ${PROJECT_SOURCE_DIR}/tool/include
        ${PROJECT_SOURCE_DIR}/source/include
        ${PROJECT_SOURCE_DIR}/math/include
        ${PROJECT_SOURCE_DIR}/model/include
        ${PROJECT_SOURCE_DIR}/Lib
        ${PROJECT_SOURCE_DIR}/Engine/include
        ${PROJECT_SOURCE_DIR}/shader/include
)
