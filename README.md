# RenderEngine / RenderGUI

一个用 C++20 编写的软件渲染引擎动态库（`RenderEngine`），提供**光栅化**与 **CUDA 路径追踪**双渲染管线，并配有基于 Qt 6（Qt Quick/QML）的跨平台图形界面（`RenderGUI`）用于加载 `.obj` 3D 模型并实时预览渲染结果。

## 项目概览

本仓库包含两个子工程：

| 子工程 | 构建产物 | 说明 |
|---|---|---|
| `RenderEngine/` | `libRenderEngine.*` (动态库) + `RenderTest` (CLI 测试程序) | 渲染引擎核心，通过 `IEngine` 纯虚接口对外暴露 |
| `GUI/` | `RenderGUI` (Qt6 GUI 应用) | 基于 Qt Quick/QML 的图形界面，动态链接 `RenderEngine` |

## 渲染特性

### 光栅化管线 (Rasterization)

- **应用阶段** — 变换指令队列处理（平移/旋转/缩放）
- **裁剪** — Cohen-Sutherland 剪裁算法
- **视口变换** — NDC 到屏幕坐标映射
- **几何着色** — Blinn-Phong 光照模型
- **光栅化** — 三角形遍历 + 重心坐标插值
- **深度测试** — 逐片元原子 ZTest
- **延迟渲染** — G-Buffer 实现
- **阴影映射** (Shadow Mapping) — 光源视角深度 Pass
- **环境光遮蔽** (SSAO)
- **抗锯齿** — 4 种模式：NOAA / FXAA / FXAAC / FXAAQ
- **伽马校正**
- **天空盒** (SkyBox)

### 路径追踪管线 (Path Tracing)

- 基于 **CUDA GPU** 加速
- Möller–Trumbore 三角形求交
- 双层 BVH 加速结构（BLAS + TLAS）
- 俄罗斯轮盘赌 (Russian Roulette)
- 余弦加权半球采样 (Cosine-weighted Hemisphere Sampling)
- BSDF 评估

## 技术栈

| 维度 | 内容 |
|---|---|
| **语言** | C++20 (主), CUDA (GPU 加速), QML/JavaScript (GUI) |
| **构建系统** | CMake >= 4.1 |
| **GUI 框架** | Qt 6 (Core, Gui, Quick, Qml, Multimedia) |
| **GPU 加速** | 可选 CUDA Toolkit (路径追踪) |
| **SIMD 优化** | x86 SSE4.1 / ARM NEON (自动检测) |
| **许可协议** | GNU GPL v3.0 |

## 项目结构

```
Project/
├── RenderEngine/                      # 渲染引擎核心
│   ├── CMakeLists.txt
│   └── Engine/
│       ├── include/                   # 引擎对外接口
│       │   ├── IEngine.hpp            #   纯虚接口 + 工厂函数 (CreateEngine/DestroyEngine)
│       │   ├── Engine.hpp             #   引擎实现类
│       │   ├── Graphic.hpp            #   图形管线
│       │   ├── GBuffer.hpp            #   延迟渲染 GBuffer
│       │   ├── ShadowMap.hpp          #   阴影映射
│       │   ├── Uniform.hpp            #   全局 Uniform 数据
│       │   └── HitInfo.hpp            #   求交信息
│       ├── src/                       # 引擎实现
│       │   ├── Pipline.cpp            #   光栅化管线主循环
│       │   ├── Engine.cpp / EngineFactory.cpp
│       │   ├── Graphic.cpp            #   三角形/片元处理
│       │   ├── AA.cpp                 #   抗锯齿
│       │   ├── AO.cpp                 #   环境光遮蔽
│       │   └── ShadowMap.cpp / GBuffer.cpp / Uniform.cpp
│       ├── CUDART/                    # CUDA 加速代码
│       │   ├── include/               #   CUDA 头文件 (.cuh)
│       │   └── src/
│       │       ├── PathTracing.cu     #     路径追踪核心 kernel
│       │       ├── BVH.cu             #     GPU BVH 遍历
│       │       ├── Shape.cu           #     三角形求交
│       │       ├── AABB.cu / HitTool.cu / Math.cu
│       │       └── Interface.cu       #     CPU↔GPU 数据传输
│       ├── math/                      # 数学库
│       │   ├── include/
│       │   │   ├── Vec.hpp            #     VecN<N> 模板向量 (SIMD 友好, alignas 对齐)
│       │   │   ├── Mat.hpp            #     MatMN<M,N> 模板矩阵
│       │   │   ├── MatPro.hpp         #     投影/视图矩阵生成
│       │   │   ├── RasterTool.hpp     #     光栅化工具 (三角形遍历)
│       │   │   ├── ClipTool.hpp       #     裁剪算法
│       │   │   ├── LerpTool.hpp       #     插值工具
│       │   │   ├── MathTool.hpp       #     欧拉角/四元数等
│       │   │   ├── GammaTool.hpp      #     伽马校正
│       │   │   └── FragTool.hpp       #     片元工具
│       │   └── src/                   #     对应 .cpp 实现
│       ├── model/                     # 模型数据结构
│       │   ├── include/ (Mesh.hpp, Shape.hpp, F2P.hpp, V2F.hpp)
│       │   └── src/ (对应实现)
│       ├── shader/                    # 着色器
│       │   ├── include/ (Shader.hpp, BlinnShader.hpp, SkyShader.hpp)
│       │   └── src/ (对应实现)
│       ├── source/                    # 场景资源
│       │   ├── include/ (Camera.hpp, Lights.hpp, Ray.hpp, Film.hpp, SkyBox.hpp, Transform.hpp, RenderObjects.hpp)
│       │   └── src/ (对应实现)
│       ├── tool/                      # 工具集
│       │   ├── include/
│       │   │   ├── BVH.hpp            #     双层 BVH (BLAS + TLAS)
│       │   │   ├── AABB.hpp           #     包围盒
│       │   │   ├── thread_pool.hpp    #     通用线程池
│       │   │   ├── SwapChain.hpp      #     双缓冲交换链
│       │   │   ├── ModelReader.hpp    #     .obj 模型加载
│       │   │   └── UVLoader.hpp       #     UV 纹理加载
│       │   └── src/ (对应实现)
│       ├── Lib/                       # 第三方库
│       │   ├── stb_image.h            #     纹理加载
│       │   └── sse2neon.h             #     x86 SSE → ARM NEON 映射
│       ├── bin/                       # 构建产物输出目录
│       └── main.cpp                   # CLI 测试程序 (RenderTest)
│
├── GUI/                               # Qt6 图形界面
│   ├── CMakeLists.txt
│   ├── Backend/                       # C++ 后端
│   │   ├── include/
│   │   │   ├── IEngine.hpp            #   引擎接口 (同步自 RenderEngine)
│   │   │   ├── SettingProxy.hpp       #   设置代理 (QObject, 暴露给 QML)
│   │   │   └── FrameProvider.hpp      #   帧数据提供 (视频输出)
│   │   └── src/
│   │       ├── main.cpp               #   应用入口
│   │       ├── SettingProxy.cpp / CameraSettings.cpp / LightSettings.cpp / GlobalSettings.cpp
│   │       └── FrameProvider.cpp
│   ├── qml/                           # QML 界面
│   │   ├── MainWindow.qml             #   主窗口
│   │   ├── widgets/TopMenuBar.qml     #   菜单栏
│   │   ├── views/
│   │   │   ├── ColorBlockColumn.qml   #   颜色设置面板
│   │   │   ├── CameraBlockColumn.qml  #   相机控制面板
│   │   │   ├── MeshBlockColumn.qml    #   网格信息面板
│   │   │   └── PosiBlockColumn.qml    #   位置变换面板
│   │   └── components/
│   │       ├── Block.qml              #   通用区块组件
│   │       ├── SetLightButton.qml     #   灯光开关按钮
│   │       └── CheckLight.qml         #   灯光状态指示
│   ├── bin/                           # 构建产物 + 运行时引擎动态库
│   └── resources.qrc
│
├── README.md
├── LICENSE
└── 重构总结.md                        # 光栅化管线重构记录
```

## 核心架构

### 引擎接口 (IEngine)

引擎通过纯虚接口 [IEngine.hpp](file:///Users/dongyu/CLionProjects/Project/RenderEngine/Engine/include/IEngine.hpp) 对外暴露，通过 C 风格工厂函数创建/销毁实例，支持跨动态库调用：

```cpp
extern "C" ENGINE_API IEngine *CreateEngine(size_t w, size_t h, bool Gamma, bool RT);
extern "C" ENGINE_API void DestroyEngine(const IEngine* engine);
```

接口覆盖：渲染模式切换（光栅化/路径追踪）、光源设置（环境光/主光源/逐像素光）、阴影/AO/AA/天空盒开关、模型加载、相机控制、变换指令、帧循环管理等。

### 渲染管线流程

```
用户操作 (GUI/QML)
    │
    ▼
SettingProxy ──→ 更新引擎设置参数 (双缓冲, 线程安全)
    │
    ▼
RenderFrame() ──→ Application()          // 应用变换指令队列
    │
    ├── 光栅化模式 ──→ Clip → ScreenMapping → GeometryShading
    │                      → Rasterize (三角形遍历 + 重心坐标)
    │                      → ZTest → WriteBuffer / WriteGBuffer
    │                      → ShadowPass → AO → AA → Gamma
    │
    └── 路径追踪模式 ──→ 数据传输到 GPU
                           → PathTracing Kernel (BVH遍历 + 求交 + 采样)
                           → 结果回读到 CPU
    │
    ▼
SwapChain ──→ IFrameReceiver::OnFrameReady() ──→ GUI 展示 / 文件保存
```

### 双层 BVH 加速结构

- **BLAS** (Bottom-Level): 每个 Mesh 独立的局部 BVH，存储三角形索引
- **TLAS** (Top-Level): 场景级全局 BVH，存储实例化引用，支持 Mesh 复用
- 支持 CPU 端递归遍历和 CUDA GPU 端并行遍历

## 构建指南

### 依赖

- CMake >= 4.1
- 支持 C++20 的编译器 (Clang / GCC / MSVC)
- Qt 6（GUI 需要）：`Core`, `Gui`, `Quick`, `Qml`, `Multimedia`
- （可选）CUDA Toolkit — 自动检测，开启路径追踪加速

### 1) 构建 RenderEngine

```bash
cd RenderEngine
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release -j
```

构建产物输出到 `RenderEngine/bin/`：

| 文件 | 说明 |
|---|---|
| `libRenderEngine.dylib` / `.so` / `.dll` | 引擎动态库 |
| `RenderTest` | CLI 测试程序 |
| `IEngine.hpp` | 导出的引擎接口头文件 |

### 2) 准备 GUI 依赖

将引擎动态库复制到 `GUI/bin/`：

```bash
# macOS
cp RenderEngine/bin/libRenderEngine.dylib GUI/bin/

# Linux
cp RenderEngine/bin/libRenderEngine.so GUI/bin/

# Windows
cp RenderEngine/bin/libRenderEngine.dll GUI/bin/
```

### 3) 构建 GUI

```bash
cd GUI
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release -j
```

## 运行

### GUI 模式 (RenderGUI)

```bash
# macOS (从 GUI/bin/ 启动 .app bundle)
open GUI/bin/RenderGUI.app

# 其他平台
./GUI/bin/RenderGUI
```

界面操作：

| 菜单 | 功能 |
|---|---|
| `File → Open` | 加载 `.obj` 模型并开始渲染预览 |
| `File → Save` | 保存当前帧（调试中，尚未完整实现） |
| `Settings → Render modes` | 切换 `Rasterization` / `Path Tracing` |
| `Settings → Rendering Settings` | 开关 AO / ShadowMapping / SkyBox / FXAA |

右侧面板提供：光源颜色调节、相机参数（FOV/近远平面）、物体旋转控制、网格统计信息。

### CLI 模式 (RenderTest)

建议在 `RenderEngine/bin/` 目录下运行：

```bash
cd RenderEngine/bin
./RenderTest
```

`main.cpp` 内置了多个测试函数，可通过修改 `main()` 中的调用切换测试场景：
- `testRas()` — 光栅化性能测试（默认）
- `testRt()` — 路径追踪渲染
- `testLight()` — 多光源场景测试
- `testAO()` — 环境光遮蔽测试
- `testSpeed()` — 路径追踪性能测试

## 性能优化

- **CUDA GPU 加速**：路径追踪管线在 GPU 上并行处理，每个线程处理一个像素
- **SIMD 指令集**：自动检测 CPU 支持 SSE4.1 或 ARM NEON 并启用
- **LTO (Link-Time Optimization)**：构建时自动检测并开启
- **线程池并行**：光栅化管线实现三角形级别的细粒度并行
- **双缓冲交换链**：避免渲染过程中的画面撕裂
- **设置双缓冲**：`SettingCache settings[2]` 实现线程安全的运行时参数热更新
- **内存优化**：逐三角形/逐片元处理模式，减少中间数据结构，提升缓存利用率

## License

GNU General Public License v3.0 — 详见 [LICENSE](./LICENSE) 文件。
