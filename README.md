# RenderEngine / RenderGUI

一个用 C++20 编写的渲染引擎动态库（`RenderEngine`），并提供基于 Qt 6（Qt Quick/QML）的简单 GUI（`RenderGUI`）用于加载 `.obj` 模型并预览渲染结果。

本仓库当前包含两个子工程：
- `RenderEngine/`：构建动态库 `RenderEngine`，并提供测试程序 `RenderTest`
- `GUI/`：Qt6 GUI 程序 `RenderGUI`，通过动态链接库调用 `RenderEngine`

## 目录结构

- `RenderEngine/`
  - `Engine/include/IEngine.hpp`：对外接口（构建后会同步到 `RenderEngine/bin/IEngine.hpp`）
  - `bin/`：构建产物与示例资源（模型、测试输出等）
- `GUI/`
  - `Backend/`：Qt 与引擎交互层（`SettingProxy` / `FrameProvider`）
  - `qml/`：界面（`MainWindow.qml` 等）
  - `bin/`：GUI 构建产物（以及运行时依赖的 `libRenderEngine.*`）

## 依赖

- CMake >= 4.1（以 `CMakeLists.txt` 中 `cmake_minimum_required(VERSION 4.1)` 为准）
- C++20 编译器
- Qt 6（GUI 需要）：`Core`, `Gui`, `Quick`, `Qml`, `Multimedia`
- （可选）CUDA Toolkit：在检测到 CUDA 编译器时，会尝试启用 CUDA 并 `find_package(CUDAToolkit REQUIRED)`

## 构建

建议分别构建 `RenderEngine` 与 `GUI`（GUI 依赖 RenderEngine 的动态库文件存在于 `GUI/bin/`）。

### 1) 构建 RenderEngine（生成动态库与 RenderTest）

```bash
cd RenderEngine
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release -j
```

构建产物默认输出到：
- 动态库：`RenderEngine/bin/libRenderEngine.*`
- 测试程序：`RenderEngine/bin/RenderTest`
- 导出头文件：`RenderEngine/bin/IEngine.hpp`

### 2) 准备 GUI 依赖的动态库

`GUI/CMakeLists.txt` 会在配置阶段检查 `GUI/bin/` 下是否存在引擎动态库文件；因此在构建 GUI 前，需要将动态库放到该位置（或自行修改 CMake 链接逻辑）。

以 macOS 为例：
```bash
cp RenderEngine/bin/libRenderEngine.dylib GUI/bin/
```

Linux / Windows 对应文件名通常为：
- Linux：`libRenderEngine.so`
- Windows：`libRenderEngine.dll`

### 3) 构建 GUI（RenderGUI）

```bash
cd GUI
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release -j
```

## 运行

### 运行 GUI（RenderGUI）

构建完成后，从 `GUI/bin/` 启动（确保同目录下存在 `libRenderEngine.*`）：
- macOS：`GUI/bin/RenderGUI.app`
- 其它平台：以 `cmake` 生成的可执行文件为准

界面中已接入的操作：
- `File -> Open`：选择 `.obj` 文件加载并开始渲染预览
- `Settings -> Render modes`：切换 `Rasterization` / `Path Tracing`
- `Settings -> Rendering Settings`：开关 `AO` / `ShadowMapping` / `SkyBox`，以及 `FXAA Strength`

注意：
- `File -> Save` 当前只会触发保存对话框并打印调试信息，尚未将渲染结果写入文件。

### 运行 RenderTest

`RenderTest` 中使用了相对路径读取模型（例如 `./model/test4.obj`）并将输出写入 `./test/`，建议在 `RenderEngine/bin/` 目录下运行：

```bash
cd RenderEngine/bin
./RenderTest
```

## 引擎接口（IEngine）

对外接口定义在 `RenderEngine/Engine/include/IEngine.hpp`（构建后会同步到 `RenderEngine/bin/IEngine.hpp`）。核心入口为：
- `CreateEngine(size_t w, size_t h, bool Gamma, bool RT)`
- `DestroyEngine(const IEngine* engine)`

以及 `IEngine` 上的一组控制与渲染方法（例如：加载网格、添加物体、灯光设置、阴影/AO/AA 开关、启动渲染循环等）。

## License

见 `LICENSE`。
