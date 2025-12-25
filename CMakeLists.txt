cmake_minimum_required(VERSION 4.1)
project(untitled)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

file(GLOB_RECURSE SRC "${PROJECT_SOURCE_DIR}/*/src/*.cpp")


add_executable(untitled ${SRC})
target_include_directories(untitled PUBLIC
        ${PROJECT_SOURCE_DIR}/source/include
        ${PROJECT_SOURCE_DIR}/math/include
        ${PROJECT_SOURCE_DIR}/model/include
)
