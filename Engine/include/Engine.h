//
// Created by 冬榆 on 2025/12/29.
//

#ifndef UNTITLED_ENGINE_H
#define UNTITLED_ENGINE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <queue>

#include "Camera.h"
#include "Film.h"
#include "Graphic.h"
#include "Uniform.h"
#include "Vec.hpp"
#include "RenderObjects.h"
#include "Lights.h"

struct TextureMap;
class Material;
struct Mesh;
class RenderObjects;
class Lights;
class Graphic;


// 变换指令(累积变换)
typedef struct TransformCommand {
    uint16_t objId{};
    enum Type { TRANSLATE, ROTATE, SCALE } type = SCALE;
    VecN<3> value;  // 移动/旋转/缩放值
} TfCmd;


class Engine {
public:
    Engine(size_t w, size_t h);
    void addTfCommand(const TransformCommand &cmd);
    std::vector<std::string> addMesh(const std::string &filename);
    uint16_t addObjects(const std::string &meshName);
    uint16_t addLight();
    void setResolution(size_t w, size_t h);
    uint16_t updateCounter();


    void Application();  // 应用物体、相机与光源的变换
    void RenderFrame(const std::vector<uint16_t>& models);  // 绘制每帧的入口，帧绘制管理
    void BeginFrame();   // 初始化帧
    void EndFrame();  // 交付帧
    void DrawScene(const std::vector<uint16_t>& models);  // 绘制场景
    friend class Graphic;

private:
    // 场景资源
    std::unordered_map<std::string, Material*> materialMap;
    std::unordered_map<std::string, TextureMap*> textureMap;
    std::unordered_map<std::string, Mesh*> meshes;

    std::unordered_map<uint16_t, RenderObjects> renderObjs;
    std::unordered_map<uint16_t, Lights> lights;
    Camera camera;

    // 变换指令队列
    std::queue<TransformCommand> tfCommand;

    size_t width, height;  // 分辨率
    Film img;
    uint16_t counter=1;  // 计数器用来返回Id，限制场景最多254个渲染对象,0号为Camera

    Graphic graphic;
    GlobalUniform globalU;  // 全局Uniform
    std::vector<float> ZBuffer;  // Z-Buffer
    Film *frontBuffer{};  // 正在显示的Buffer
    Film *backBuffer{};  // 正在绘制的Buffer
};


#endif //UNTITLED_ENGINE_H