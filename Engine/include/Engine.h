//
// Created by 冬榆 on 2025/12/29.
//

#ifndef UNTITLED_ENGINE_H
#define UNTITLED_ENGINE_H

#include "Camera.h"
#include "GBuffer.h"
#include "Graphic.h"
#include "Uniform.h"
#include "Lights.h"
#include "ShadowMap.h"


// ID体系:0号为Camera,1号为主光源,2-4号为逐片元光源
enum sysID : size_t{CameraID, MainLightID, PixL1, PixL2, PixL3, RenderObject, VexLight, Error};

// 变换指令(累积变换)
typedef struct TransformCommand {
    size_t objId{};
    sysID typeId{};
    enum Type { TRANSLATE, ROTATE, SCALE } type = SCALE;
    VecN<3> value;  // 移动/旋转/缩放值
} TfCmd;


class Engine {
public:
    Engine(size_t w, size_t h);
    ~Engine();
    void SetMainLight(size_t w, size_t h);
    void SetEnvLight(uint8_t r, uint8_t g, uint8_t b, float I);
    void CloseShadow() { NeedShadowPass = false; }
    void OpenShadow() { NeedShadowPass = true; }
    void addTfCommand(const TransformCommand &cmd);
    std::vector<std::string> addMesh(const std::string &filename);
    size_t addObjects(const std::string &meshName);
    sysID addPixLight(Lights &light);
    size_t addVexLight(Lights &light);
    void setResolution(size_t w, size_t h);


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
    std::unordered_map<std::string, TextureMap*> bumpMap;
    std::unordered_map<std::string, Mesh*> meshes;

    std::vector<RenderObjects> renderObjs;
    std::array<Lights, 3> PixLights;  // 主要光源(逐像素光源)
    std::vector<Lights> VexLights;  // 次要光源(顶点光源)
    Camera camera;
    MainLight *mainLight{};
    EnvironmentLight *envLight{};

    // 变换指令队列
    std::queue<TransformCommand> tfCommand;
    bool NeedShadowPass = false;  // 是否需要阴影Pass

    size_t width, height;  // 分辨率
    Film img;

    Graphic graphic;
    GlobalUniform globalU;  // 全局Uniform
    std::vector<float> ZBuffer;  // Z-Buffer
    ShadowMap ShadowMap;  // 阴影Z-Buffer
    GBuffer GBuffer;  // GBuffer
    Film *frontBuffer{};  // 正在显示的Buffer
    Film *backBuffer{};  // 正在绘制的Buffer
};


#endif //UNTITLED_ENGINE_H