//
// Created by 冬榆 on 2025/12/29.
//

#include <ranges>
#include <thread>

#include "Engine.hpp"
#include "F2P.hpp"
#include "GammaTool.hpp"
#include "Mesh.hpp"
#include "ModelReader.hpp"
#include "RenderObjects.hpp"

Engine::Engine(const size_t w, const size_t h, const bool Gamma, const bool RT)
    : width(w)
    , height(h)
    , img(w, h)
    , graphic(this)
    , globalU(w, h, w, h) {
    // 预备分配缓冲
    SdMap = std::make_shared<ShadowMap>(w, h);
    ZBuffer.resize(w * h, 1.0f);
    tmpBufferF.resize(w * h, FloatPixel{0.0f, 0.0f, 0.0f, 0.0f});
    tmpBufferB.resize(w * h, FloatPixel{0.0f, 0.0f, 0.0f, 0.0f});
    Engine::CloseShadow(); // 默认关闭阴影
    VexLights.clear();
    NeedGammaCorrection = Gamma;
    gBuffer = std::make_unique<GBuffer>(w, h);
    const float aspect = static_cast<float>(w) / static_cast<float>(h);
    camera.setAsp(aspect);
    if (mainLight) mainLight->setAsp(aspect);
    settings[1].IsRT = RT;
    swapChain = std::make_unique<SwapChain>(w, h);
}

Engine::~Engine() {
    delete mainLight;
    envLight = nullptr;
    Engine::stopLoop();
}

// 设置主光源，并更新shadow map分辨率，需要光源与阴影贴图分辨率,不会更改阴影开关
void Engine::SetMainLight(const uint8_t r, const uint8_t g, const uint8_t b, const float I) {
    if (!mainLight) mainLight = new MainLight();
    mainLight->intensity = I;
    mainLight->color = Pixel{r, g, b};
    SdMap->resize(width, height);
    globalU.setShadowViewPort(width, height);
}

// 设置全局环境光
void Engine::SetEnvLight(const uint8_t r, const uint8_t g, const uint8_t b, const float I) {
    if (!envLight) envLight = new EnvironmentLight();
    envLight->setColor(r, g, b, 255);
    envLight->setI(I);
}

// 直接设置逐像素光源，不校验是否有效
void Engine::SetPixLight(const sysID plId, const uint8_t r, const uint8_t g, const uint8_t b, float I) {
    PixLights[plId - PixL1].setColor(r, g, b);
}

// 添加变换指令到队列中
void Engine::addTfCommand(const TfCmd &cmd) {
    tfCommand.push(cmd);
}

void Engine::addTfCommand(const size_t objId, const sysID typeId, const TfType Ttype, std::array<float, 3> value) {
    addTfCommand({objId, typeId, Ttype, {value[0], value[1], value[2]}});
}

// 添加网格模型，返回网格名字
std::vector<std::string> Engine::addMesh(const std::string &filename) {
    auto MiD = ModelReader::readObjFile(
        NeedGammaCorrection, filename, meshes, materialMap, textureMap, normalMap);
    for (const auto& mID : MiD) {
        const auto mesh = meshes.at(mID);
        if (mesh->BLASIdx != -1) continue;
        mesh->BLASIdx = static_cast<int>(blasList.size());
        auto blas = mesh->BuildBLAS(mesh);
        blasList.emplace_back(blas);
    }
    return MiD;
}

// 添加渲染物体,输入物体网格名字,返回物品ID
size_t Engine::addObjects(const std::string &meshName) {
    RenderObjects obj(meshes.at(meshName));
    const auto updateCounter = renderObjs.size();
    this->renderObjs.push_back(std::move(obj));
    return updateCounter;
}

// 添加逐片元灯光，返回灯光索引。当i = 3时表示添加失败
sysID Engine::addPixLight(const uint8_t r, const uint8_t g, const uint8_t b, const LType type) {
    for (int id_val = PixL1; id_val <= PixL3; ++id_val) {
        if (const auto id = static_cast<sysID>(id_val); !PixLights[id-PixL1].alive) {
            auto& light = PixLights[id-PixL1];
            light.alive = true;
            light.setColor(r, g, b);
            light.setI(5.0f);
            light.setLtype(type);
            return id;;
        }
    }
    return Error;
}

// 添加逐顶点灯光，返回灯光ID。
size_t Engine::addVexLight(const uint8_t r, const uint8_t g, const uint8_t b, const LType type) {
    Lights light;
    light.alive = true;
    light.setColor(r, g, b);
    light.setI(5.0f);
    light.setLtype(type);
    // VexLights.clear();
    const auto updateCounter = VexLights.size();
    VexLights.push_back(light);
    return updateCounter;
}

// 设置渲染分辨率，重置显示胶片
void Engine::setResolution(const size_t w, const size_t h) {
    width = w;
    height = h;
    swapChain = std::make_unique<SwapChain>(w, h);
    ZBuffer.resize(w * h, 1.0f);
    tmpBufferF.resize(w * h, FloatPixel{0.0f,0.0f,0.0f,0.0f});
    tmpBufferB.resize(w * h, FloatPixel{0.0f,0.0f,0.0f,0.0f});
    globalU = GlobalUniform(w, h, w, h);
    // 注意还需要修正相机和灯光的Aspect
    const float aspect = static_cast<float>(w) / static_cast<float>(h);
    camera.setAsp(aspect);
    mainLight->setAsp(aspect);
}

// 绘制场景-前向渲染
void Engine::DrawScene(const std::vector<uint16_t>& models) {
    const bool NeedSkyBoxPass = settings[0].NeedSkyBoxPass;
    const bool NeedShadowPass = settings[0].NeedShadowPass;
    // SkyPass
    Mat4 PV;
    if (NeedSkyBoxPass) {
        PV = Transpose(camera.RMat()) * camera.invProjectionMat();
        const auto uniform = Uniform(sky.ModelMat(), sky.updateMVP(PV),
               sky.InverseTransposedMat());
        graphic.SkyPass(sky, uniform, globalU, 0);
    }

    // ShadowPass
    if (NeedShadowPass && mainLight != nullptr) {
        PV = mainLight->ProjectionMat() * mainLight->ViewMat();
        globalU.setProjectViewShadow(PV);  // 更新全局Uniform光源相关内容
        for (const auto& model : models) {
            auto obj = renderObjs.at(model);
            auto uniform = Uniform(obj.ModelMat(), obj.updateMVP(PV),
                           obj.InverseTransposedMat());
            graphic.ShadowPass(obj, uniform, globalU, 1);
        }
    }

    // BasePass
    PV = camera.ProjectionMat() * camera.ViewMat();
    globalU.setCameraViewM(camera.ViewMat());
    globalU.setCameraProjM(camera.ProjectionMat());
    // 更新全局Uniform相机相关内容
    globalU.setCameraPos(camera.getPosi());
    for (const auto& model : models) {
        auto obj = renderObjs.at(model);
        auto uniform = Uniform(obj.ModelMat(),
                                obj.updateMVP(PV),
                        obj.InverseTransposedMat());
        graphic.BasePass(obj, uniform, globalU, 2);
    }
}

void Engine::DrawScenceRT(const std::vector<uint16_t>& models) {
    for (auto& model : models) {
        auto& renderObj = renderObjs.at(model);
        renderObj.ModelMat();  // 更新M
    }
    graphic.RT(settings[0].SSP);
}

// 后处理阶段,工作集中于tmpBuffer
void Engine::PostProcess() {
    const auto aaType = settings[0].aaType;
    // 环境光遮蔽(光线追踪时不起作用)
    if (settings[0].NeedAo && !settings[0].IsRT) {
        std::ranges::fill(tmpBufferB, FloatPixel{0.0f,0.0f,0.0f,0.0f});
        graphic.SSAO(tmpBufferF, tmpBufferB, gBuffer->Gdata, ZBuffer, globalU.getCameraView(), globalU.getCameraProj());
        std::swap(tmpBufferF, tmpBufferB);
    }

    if (aaType == NOAA) return;
    std::ranges::fill(tmpBufferB, FloatPixel{0.0f,0.0f,0.0f,0.0f});
    if (aaType == FXAA) {
        graphic.FXAA(tmpBufferF, tmpBufferB);
    } else if (aaType == FXAAC) {
        graphic.FXAAC(tmpBufferF, tmpBufferB);
    } else if (aaType == FXAAQ) {
        graphic.FXAAQ(tmpBufferF, tmpBufferB);
    }
    std::swap(tmpBufferF, tmpBufferB);
}

// 帧绘制管理
void Engine::RenderFrame(const std::vector<uint16_t>& models) {
    BeginFrame();   // 初始化帧
    if (rendBuffer) {
        Application();  // 应用变换
        // 绘制指定models
        if (settings[0].IsRT) {
            BuildTLAS(models);// 光线追踪需要初始化BVH
            DrawScenceRT(models);
        } else DrawScene(models);
        PostProcess();   // 画面后处理
        EndFrame();      // 交付帧
    }
}

void Engine::BeginFrame() {
    // 清空ZBuffer
    std::ranges::fill(ZBuffer, 1.0f);
    std::ranges::fill(tmpBufferF, FloatPixel{0.0f,0.0f,0.0f,0.0f});
    std::ranges::fill(tmpBufferB, FloatPixel{0.0f,0.0f,0.0f,0.0f});
    SdMap->clear();
    rendBuffer = swapChain->acquireBackBuffer();
    if (rendBuffer) rendBuffer->clear();
    gBuffer = std::make_unique<GBuffer>(width, height);

    // 同步 设置双缓冲
    {
        std::lock_guard lock(settingMtx);
        memcpy(&settings[0], &settings[1], sizeof(SettingCache));
    }
    camera.setFOV(settings[0].fov);
    camera.setNear(settings[0].near);
    camera.setFar(settings[0].far);
}

void Engine::EndFrame() {
    // 处理缓冲区的像素转到图像缓冲区
    Write2Front();
    // rendBuffer->save("test.pam");
    swapChain->commitBackBuffer(rendBuffer);
    rendBuffer = nullptr;
}

// 写入绘制缓冲区,自行转换伽马矫正
void Engine::Write2Front() {
    const auto totalPixels = tmpBufferF.size();
    const auto threadCount = std::thread::hardware_concurrency();
    const int chunkSize = std::max(1, static_cast<int>((totalPixels + threadCount - 1) / threadCount));
    
    if (!NeedGammaCorrection) {
        std::vector<std::future<void>> futures;
        for (int startIdx = 0; startIdx < totalPixels; startIdx += chunkSize) {
            const int endIdx = std::min(startIdx + chunkSize, static_cast<int>(totalPixels));
            futures.emplace_back(pool.addTask([&, startIdx, endIdx]() {
                for (auto i = startIdx; i < endIdx; ++i) {
                    rendBuffer->image[i] = tmpBufferF[i].toPixel();
                }
            }));
        }
        for (auto& future : futures) {
            future.wait();
        }
        return;
    }
    
    std::vector<std::future<void>> futures;
    for (int startIdx = 0; startIdx < totalPixels; startIdx += chunkSize) {
        const int endIdx = std::min(startIdx + chunkSize, static_cast<int>(totalPixels));
        futures.emplace_back(pool.addTask([&, startIdx, endIdx]() {
            for (auto i = startIdx; i < endIdx; ++i) {
                auto& pix = tmpBufferF[i];
                pix.r = linearToSrgb(pix.r);
                pix.g = linearToSrgb(pix.g);
                pix.b = linearToSrgb(pix.b);
                rendBuffer->image[i] = pix.toPixel();
            }
        }));
    }
    for (auto& future : futures) {
        future.wait();
    }
}

std::array<size_t, 2> Engine::getTriVexNums() {
    size_t TriNums = 0;
    size_t VexNums = 0;
    for (const auto& rObj : renderObjs) {
        const auto& mesh = rObj.getMesh();
        TriNums += mesh->getTriNums();
        VexNums += mesh->getVBONums();
    }
    return {TriNums, VexNums};
}

void Engine::startLoop(std::vector<uint16_t> objs, IFrameReceiver* receiver) {
    if (renderLoop || pullLoop) return;
    if (!swapChain) swapChain = std::make_unique<SwapChain>(width, height);
    renderLoop = true;
    pullLoop = true;

    RendWorker = std::thread([this, objs = std::move(objs)] {
        while (renderLoop) {
            RenderFrame(objs);
        }
    });

    PullWorker = std::thread([this, receiver] {
        while (pullLoop) {
            const auto frame = swapChain->acquireFrontBuffer();
            if (!frame) break;  // 当返回空时交换链被关闭
            if (receiver) {
                // 接收器拿到缓冲区数据进行显示等操作
                receiver->OnFrameReady(frame->image.data());
            }
            swapChain->releaseFrontBuffer(frame);  // 归还缓冲区
        }
    });
}

void Engine::stopLoop() {
    renderLoop = false;
    pullLoop = false;
    if (swapChain) {
        swapChain->stop();
        swapChain = std::make_unique<SwapChain>(width, height);
    }
    if (RendWorker.joinable()) RendWorker.join();
    if (PullWorker.joinable()) PullWorker.join();
}
