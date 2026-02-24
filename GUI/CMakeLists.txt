cmake_minimum_required(VERSION 4.1)
project(RenderGUI CXX)

set(CMAKE_CXX_STANDARD 20)
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

file(GLOB_RECURSE SRC "${PROJECT_SOURCE_DIR}/*/src/*.cpp")
file(GLOB_RECURSE SRC_H "${PROJECT_SOURCE_DIR}/*/include/*.hpp")
find_package(Qt6 REQUIRED COMPONENTS Core Gui Quick Qml Multimedia)

qt_add_executable(RenderGUI
        WIN32 MACOSX_BUNDLE
        ${SRC} ${SRC_H}
        resources.qrc
)

target_link_libraries(RenderGUI PRIVATE Qt6::Core Qt6::Gui Qt6::Quick Qt6::Qml Qt6::Multimedia)
# 链接动态库
if(APPLE)
    set(RENDER_ENGINE_LIB "${PROJECT_SOURCE_DIR}/bin/libRenderEngine.dylib")
elseif(WIN32)
    set(RENDER_ENGINE_LIB "${PROJECT_SOURCE_DIR}/bin/libRenderEngine.dll")
else()
    set(RENDER_ENGINE_LIB "${PROJECT_SOURCE_DIR}/bin/libRenderEngine.so")
endif()
if(EXISTS ${RENDER_ENGINE_LIB})
    target_link_libraries(RenderGUI PRIVATE ${RENDER_ENGINE_LIB})
else()
    message(FATAL_ERROR "RenderEngine library not found at: ${RENDER_ENGINE_LIB}")
endif()

target_include_directories(RenderGUI PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Backend/include
)