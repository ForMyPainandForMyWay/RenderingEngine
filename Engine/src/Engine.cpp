//
// Created by 冬榆 on 2025/12/29.
//

#include "Engine.h"
#include "Graphic.h"
#include "ModelReader.h"
#include "RenderObjects.h"
#include "Lights.h"

Engine::Engine(const size_t w, const size_t h)
    : width(w)
    , height(h)
    , img(w, h)
    , graphic(this)
    , globalU(w, h)
    , frontBuffer(new Film(w, h))
    , backBuffer(new Film(w, h))
{ ZBuffer.resize(w * h, std::numeric_limits<float>::infinity());}  // 预备分配缓冲

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
    std::ranges::fill(ZBuffer, std::numeric_limits<float>::infinity());
    globalU = GlobalUniform(w, h);
}

// 更新并返回Counter计数，加入实例前必须调用
uint16_t Engine::updateCounter() {
    return ++counter;
}

// 绘制场景
void Engine::DrawScene(const std::vector<uint16_t>& models) {
    const auto PV = camera.ProjectionMat() * camera.ViewMat();
    globalU.setProjectView(PV);  // 更新全局Uniform
    for (const auto& model : models) {
        auto obj = renderObjs.at(model);
        auto uniform = Uniform(obj.updateMVP(PV),
                       obj.InverseTransposedMat());
        graphic.DrawModel(obj, uniform, 0);
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
    std::ranges::fill(ZBuffer, std::numeric_limits<float>::infinity());
    // 清空backBuffer
    backBuffer->clear();
}

void Engine::EndFrame() {
    Film* tmp = frontBuffer;
    frontBuffer = backBuffer;
    backBuffer = tmp;
    frontBuffer->save("test.pam");
}
