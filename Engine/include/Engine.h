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
#include "Vec.hpp"

class Material;
struct TextureMap;
class Mesh;
class RenderObjects;
class Lights;
class Graphic;


// 变换指令(累积变换)
typedef struct TransformCommand {
    uint8_t objId{};
    enum Type { TRANSLATE, ROTATE, SCALE } type = SCALE;
    VecN<3> value;  // 移动/旋转/缩放值
} TfCmd;


class Engine {
public:
    Engine(size_t w, size_t h);
    void addTfCommand(const TransformCommand &cmd);
    void addMesh(const std::string &filename);
    uint8_t addObjects(const std::string &meshName);
    uint8_t addLight();
    void setResolution(size_t w, size_t h);
    uint8_t updateCounter();


    void Application();  // 应用物体、相机与光源的变换
    void DrawScene(const std::vector<uint8_t>& models);  // 绘制场景
    // void VertexShader();  // 顶点处理
    // void PrimitiveAssembly();  // 图元组装、裁剪
    // void ClipProcess();   // 视锥剔除+顶点裁剪+透视除法
    // void ScreenMapping();  // 视口变换
    // void SegmentShader
    // void Rasterization();   // 光栅化

    friend class Graphic;

private:
    // 场景资源
    std::unordered_map<std::string, Material*> materialMap;
    std::unordered_map<std::string, TextureMap*> textureMap;
    std::unordered_map<std::string, Mesh*> meshes;

    std::unordered_map<uint8_t, RenderObjects> renderObjs;
    std::unordered_map<uint8_t, Lights> lights;
    Camera camera;

    // 变换指令队列
    std::queue<TransformCommand> tfCommand;

    size_t width, height;  // 分辨率
    Film img;
    uint8_t counter=1;  // 计数器用来返回Id，限制场景最多254个渲染对象,0号为Camera

    Graphic graphic;
};


#endif //UNTITLED_ENGINE_H