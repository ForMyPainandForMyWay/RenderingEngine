cmake_minimum_required(VERSION 4.1)
project(RenderGUI CXX)

set(CMAKE_CXX_STANDARD 20)
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

file(GLOB_RECURSE SRC "${PROJECT_SOURCE_DIR}/*/src/*.cpp")
file(GLOB_RECURSE SRC_H "${PROJECT_SOURCE_DIR}/*/include/*.hpp")
find_package(Qt6 REQUIRED COMPONENTS Core Gui Quick Qml)

qt_add_executable(RenderGUI
        WIN32 MACOSX_BUNDLE
        ${SRC} ${SRC_H}
        resources.qrc
)

target_link_libraries(RenderGUI PRIVATE Qt6::Core Qt6::Gui Qt6::Quick Qt6::Qml)
# 链接动态库
target_link_libraries(RenderGUI PRIVATE ${PROJECT_SOURCE_DIR}/bin/libRenderEngine.dylib)

target_include_directories(RenderGUI PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Backend/include
)