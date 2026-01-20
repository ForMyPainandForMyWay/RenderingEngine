//
// Created by 冬榆 on 2025/12/29.
//

#include <ranges>

#include "Engine.h"

#include "GammaTool.h"
#include "ModelReader.h"
#include "RenderObjects.h"

Engine::Engine(const size_t w, const size_t h, bool Gamma)
    : width(w)
    , height(h)
    , img(w, h)
    , graphic(this)
    , globalU(w, h, w, h)
    , ShadowMap(w, h)
    , frontBuffer(new Film(w, h))
    , backBuffer(new Film(w, h)) {
    // 预备分配缓冲
    ZBuffer.resize(w * h, 1.0f);
    tmpBufferF.resize(w * h, FloatPixel{0.0f, 0.0f, 0.0f, 0.0f});
    tmpBufferB.resize(w * h, FloatPixel{0.0f, 0.0f, 0.0f, 0.0f});
    CloseShadow(); // 默认关闭阴影
    VexLights.clear();
    NeedGammaCorrection = Gamma;
    gBuffer = std::make_unique<GBuffer>(w, h);
}

Engine::~Engine() {
    delete backBuffer;
    delete frontBuffer;
    delete mainLight;
    delete envLight;
}

// 设置主光源，并更新shadow map分辨率，需要光源与阴影贴图分辨率,不会更改阴影开关
void Engine::SetMainLight() {
    delete mainLight;
    mainLight = new MainLight();  // 先不进行详细参数设置
    mainLight->setI(5.0f);
    ShadowMap.resize(width, height);
    globalU.setShadowViewPort(width, height);
}

// 设置全局环境光
void Engine::SetEnvLight(const uint8_t r, const uint8_t g, const uint8_t b, const float I) {
    delete envLight;
    envLight = new EnvironmentLight();
    envLight->setColor(r, g, b, 255);
    envLight->setI(I);
}

// 添加变换指令到队列中
void Engine::addTfCommand(const TransformCommand &cmd) {
    tfCommand.push(cmd);
}

// 添加网格模型，返回网格名字
std::vector<std::string> Engine::addMesh(const std::string &filename) {
    auto MiD = ModelReader::readObjFile(
        NeedGammaCorrection, filename, meshes, materialMap, textureMap, bumpMap);
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
sysID Engine::addPixLight(Lights &light) {
    light.alive = true;
    for (int id_val = PixL1; id_val <= PixL3; ++id_val) {
        if (const auto id = static_cast<sysID>(id_val); !PixLights[id-PixL1].alive) {
            PixLights[id-PixL1] = light;
            return id;
        }
    }
    return Error;
}

// 添加逐顶点灯光，返回灯光ID。
size_t Engine::addVexLight(Lights &light) {
    light.alive = true;
    VexLights.clear();
    const auto updateCounter = VexLights.size();
    VexLights.push_back(std::move(light));
    return updateCounter;
}

// 设置渲染分辨率，重置显示胶片
void Engine::setResolution(const size_t w, const size_t h) {
    width = w;
    height = h;
    delete frontBuffer;
    delete backBuffer;
    frontBuffer = new Film(w, h);
    backBuffer  = new Film(w, h);
    ZBuffer.resize(w * h, 1.0f);
    tmpBufferF.resize(w * h, FloatPixel{0.0f,0.0f,0.0f,0.0f});
    tmpBufferB.resize(w * h, FloatPixel{0.0f,0.0f,0.0f,0.0f});
    globalU = GlobalUniform(w, h, w, h);

}

// 绘制场景-前向渲染
void Engine::DrawScene(const std::vector<uint16_t>& models) {
    // SkyPass
    MatMN<4, 4> PV;
    if (NeedSkyBoxPass) {
        PV = camera.RMat().Transpose() * camera.invProjectionMat();
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
    // globalU.setProjectView(PV);  // 更新全局Uniform相机相关内容
    globalU.setCameraPos(camera.getPosi());
    for (const auto& model : models) {
        auto obj = renderObjs.at(model);
        auto uniform = Uniform(obj.ModelMat(),
                                obj.updateMVP(PV),
                        obj.InverseTransposedMat());
        graphic.BasePass(obj, uniform, globalU, 2);
    }
}

// 后处理阶段,工作集中于tmpBuffer
void Engine::PostProcess() {
    // 环境光遮蔽
    if (NeedAo) {
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
    Application();  // 应用变换
    DrawScene(models);  // 绘制指定models
    PostProcess();   // 画面后处理
    EndFrame();      // 交付帧
}

void Engine::BeginFrame() {
    // 清空ZBuffer
    std::ranges::fill(ZBuffer, 1.0f);
    std::ranges::fill(tmpBufferF, FloatPixel{0.0f,0.0f,0.0f,0.0f});
    std::ranges::fill(tmpBufferB, FloatPixel{0.0f,0.0f,0.0f,0.0f});
    ShadowMap.clear();    // 清空backBuffer
    backBuffer->clear();
    gBuffer = std::make_unique<GBuffer>(width, height);
}

void Engine::EndFrame() {
    // 处理缓冲区的像素转到图像缓冲区
    Write2Front();
    std::swap(frontBuffer, backBuffer);
    frontBuffer->save("test.pam");
}

// 写入绘制缓冲区,自行转换伽马矫正
void Engine::Write2Front() {
    if (!NeedGammaCorrection) {
        for (auto i = 0; i < tmpBufferF.size(); ++i) {
            backBuffer->image[i] = tmpBufferF[i].toPixel();
        }
        return;
    }
    for (auto i = 0; i < tmpBufferF.size(); ++i) {
        auto& pix = tmpBufferF[i];
        pix.r = linearToSrgb(pix.r);
        pix.g = linearToSrgb(pix.g);
        pix.b = linearToSrgb(pix.b);
        backBuffer->image[i] = pix.toPixel();
    }
}