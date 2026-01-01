//
// Created by 冬榆 on 2025/12/29.
//

#include "Engine.h"
#include "Graphic.h"
#include "ModelReader.h"
#include "RenderObjects.h"
#include "Lights.h"

Engine::Engine(const size_t w, const size_t h) : img(w, h), graphic(this, nullptr) {
    this->width = w;
    this->height = h;
}

// 添加变换指令到队列中
void Engine::addTfCommand(const TransformCommand &cmd) {
    this->tfCommand.push(cmd);
}

// 添加网格模型
void Engine::addMesh(const std::string &filename) {
    ModelReader::readObjFile(filename, meshes, materialMap, textureMap);
}

// 添加渲染物体,返回物品ID
uint8_t Engine::addObjects(const std::string &meshName) {
    RenderObjects obj(meshes[meshName]);
    this->renderObjs.try_emplace(updateCounter(), std::move(obj));
    return counter;
}

uint8_t Engine::addLight() {
    Lights light{};
    this->lights.try_emplace(updateCounter(), std::move(light));
    return counter;
}

// 设置渲染分辨率，重置显示胶片
void Engine::setResolution(const size_t w, const size_t h) {
    this->width = w;
    this->height = h;
    img = Film(w, h);
}

// 更新并返回Counter计数，加入实例前必须调用
uint8_t Engine::updateCounter() {
    return ++counter;
}

