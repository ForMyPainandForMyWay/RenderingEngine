//
// Created by 冬榆 on 2025/12/29.
//

#include "Engine.h"

#include <ranges>

#include "Graphic.h"
#include "ModelReader.h"
#include "RenderObjects.h"
#include "Lights.h"
#include "Mesh.h"

Engine::Engine(const size_t w, const size_t h)
    : width(w)
    , height(h)
    , img(w, h)
    , graphic(this)
    , globalU(w, h, w, h)
    , ShadowMap(w, h)
    // , GBuffer(w, h)
    , frontBuffer(new Film(w, h))
    , backBuffer(new Film(w, h)) {
    // 预备分配缓冲
    ZBuffer.resize(w * h, 1.0f);
    CloseShadow();
}

Engine::~Engine() {
    delete backBuffer;
    delete frontBuffer;
    if (mainLight != nullptr) delete mainLight;
    if (envLight != nullptr) delete envLight;
    for (const auto &mat: materialMap | std::views::values) {
        delete mat;
    }
    for (const auto &tex: textureMap | std::views::values) {
        delete tex;
    }
    for (const auto &mesh: meshes | std::views::values) {
        delete mesh;
    }
}

// 设置主光源，并更新shadow map分辨率，需要光源与阴影贴图分辨率,不会更改阴影开关
void Engine::SetMainLight(const size_t w, const size_t h) {
    if (mainLight != nullptr) delete mainLight;
    mainLight = new MainLight();  // 先不进行详细参数设置
    ShadowMap.resize(w, h);
    globalU.setShadowViewPort(w, h);
}

// 设置全局环境光
void Engine::SetEnvLight(const uint8_t r, const uint8_t g, const uint8_t b, const float I) {
    if (envLight != nullptr) delete envLight;
    envLight = new EnvironmentLight();
    envLight->setColor(r, g, b, 255);
    envLight->setI(I);
}

// 添加变换指令到队列中
void Engine::addTfCommand(const TransformCommand &cmd) {
    this->tfCommand.push(cmd);
}

// 添加网格模型，返回网格名字
std::vector<std::string> Engine::addMesh(const std::string &filename) {
    auto MiD = ModelReader::readObjFile(filename, meshes, materialMap, textureMap);
    return MiD;
}

// 添加渲染物体,输入物体网格名字,返回物品ID
uint16_t Engine::addObjects(const std::string &meshName) {
    RenderObjects obj(meshes.at(meshName));
    this->renderObjs.try_emplace(updateCounter(), std::move(obj));
    return counter;
}

uint16_t Engine::addLight() {
    Lights light{};
    this->lights.try_emplace(updateCounter(), std::move(light));
    return counter;
}

// 设置渲染分辨率，重置显示胶片
void Engine::setResolution(const size_t w, const size_t h) {
    width = w;
    height = h;
    delete frontBuffer;
    delete backBuffer;
    frontBuffer = new Film(w, h);
    backBuffer  = new Film(w, h);
    ZBuffer.resize(w * h);
    std::ranges::fill(ZBuffer, 1.0f);
    globalU = GlobalUniform(w, h, w, h);
}

// 更新并返回Counter计数，加入实例前必须调用
uint16_t Engine::updateCounter() {
    return ++counter;
}

// 绘制场景-前向渲染
void Engine::DrawScene(const std::vector<uint16_t>& models) {
    // ShadowPass
    MatMN<4, 4> PV;
    if (NeedShadowPass && mainLight != nullptr) {
        PV = mainLight->ProjectionMat() * mainLight->ViewMat();
        globalU.setProjectViewShadow(PV);  // 更新全局Uniform光源相关内容
        for (const auto& model : models) {
            auto obj = renderObjs.at(model);
            auto uniform = Uniform(obj.ModelMat(), obj.updateMVP(PV),
                           obj.InverseTransposedMat());
            graphic.ShadowPass(obj, uniform, globalU, 0);
        }
    }

    // BasePass
    PV = camera.ProjectionMat() * camera.ViewMat();
    globalU.setProjectView(PV);  // 更新全局Uniform相机相关内容
    globalU.setCameraPos(camera.getPosi());
    for (const auto& model : models) {
        auto obj = renderObjs.at(model);
        auto uniform = Uniform(obj.ModelMat(),
                                obj.updateMVP(PV),
                        obj.InverseTransposedMat());
        graphic.BasePass(obj, uniform, globalU, 1);
    }
}

// 帧绘制管理
void Engine::RenderFrame(const std::vector<uint16_t>& models) {
    BeginFrame();   // 初始化帧
    Application();  // 应用变换
    DrawScene(models);  // 绘制指定models
    // PostProcess();   // 画面后处理
    EndFrame();      // 交付帧
}

void Engine::BeginFrame() {
    // 清空ZBuffer
    std::ranges::fill(ZBuffer, 1.0f);
    ShadowMap.clear();    // 清空backBuffer
    backBuffer->clear();
    // GBuffer.clear();  // 前向渲染时无需GBuffer
}

void Engine::EndFrame() {
    Film* tmp = frontBuffer;
    frontBuffer = backBuffer;
    backBuffer = tmp;
    frontBuffer->save("test.pam");
}
