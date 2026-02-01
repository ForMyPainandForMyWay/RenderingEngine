//
// Created by 冬榆 on 2025/12/29.
//

#ifndef UNTITLED_ENGINE_H
#define UNTITLED_ENGINE_H

#include <queue>

#include "BVH.hpp"
#include "Camera.hpp"
#include "GBuffer.hpp"
#include "Graphic.hpp"
#include "Uniform.hpp"
#include "Lights.hpp"
#include "ShadowMap.hpp"
#include "SkyBox.hpp"
#include "thread_pool.hpp"


struct BLAS;
class ThreadPool;

// ID体系:0号为Camera,1号为主光源,2-4号为逐片元光源
enum sysID : size_t{CameraID, MainLightID, PixL1, PixL2, PixL3, RenderObject, VexLight, Error};
enum AATpye{NOAA, FXAA, FXAAC, FXAAQ};  // 抗锯齿效果

// 变换指令(累积变换)
typedef struct TransformCommand {
    size_t objId{};
    sysID typeId{};
    enum Type { TRANSLATE, ROTATE, SCALE } type = SCALE;
    Vec3 value;  // 移动/旋转/缩放值
} TfCmd;

class Engine {
public:
    Engine(size_t w, size_t h, bool Gamma=false, bool RT=false);
    ~Engine();
    void SetMainLight();
    void SetEnvLight(uint8_t r, uint8_t g, uint8_t b, float I);
    void CloseShadow() { NeedShadowPass = false; }
    void OpenShadow() { NeedShadowPass = true; }
    void CloseSky() { NeedSkyBoxPass = false; }
    void OpenSky() { NeedSkyBoxPass = true; }
    void SetAA(const AATpye type){ aaType = type; }
    void OpenAO() { NeedAo = true; }
    void addTfCommand(const TransformCommand &cmd);
    std::vector<std::string> addMesh(const std::string &filename);
    size_t addObjects(const std::string &meshName);
    sysID addPixLight(Lights &light);
    size_t addVexLight(Lights &light);
    void setResolution(size_t w, size_t h);
    // 光线追踪
    void BuildTLAS(const std::vector<uint16_t>& models);
    std::optional<HitInfo> GetClosestHit(const Ray &worldRay) const;


    void Application();  // 应用物体、相机与光源的变换
    void RenderFrame(const std::vector<uint16_t>& models);  // 绘制每帧的入口，帧绘制管理
    void BeginFrame();   // 初始化帧
    void EndFrame();  // 交付帧
    void DrawScene(const std::vector<uint16_t>& models);  // 绘制场景
    void DrawScenceRT(const std::vector<uint16_t>& models);  // 光线追踪路径
    void PostProcess();  // 后处理
    void Write2Front();  // 写入绘制缓冲区(交付),自行转换伽马矫正

    friend class Graphic;

private:
    // 场景资源
    std::unordered_map<std::string, std::shared_ptr<Material>> materialMap;
    std::unordered_map<std::string, std::shared_ptr<TextureMap>> textureMap;
    std::unordered_map<std::string, std::shared_ptr<TextureMap>> normalMap;
    std::unordered_map<std::string, std::shared_ptr<Mesh>> meshes;
    std::vector<std::shared_ptr<BLAS>> blasList;
    std::unique_ptr<TLAS> tlas;
    SkyBox sky;
    std::vector<RenderObjects> renderObjs;
    std::array<Lights, 3> PixLights;  // 主要光源(逐像素光源)
    std::vector<Lights> VexLights;  // 次要光源(顶点光源)
    Camera camera;
    MainLight *mainLight{};
    EnvironmentLight *envLight{};

    // 变换指令队列
    std::queue<TransformCommand> tfCommand;
    bool IsRT;  // 是否是光线追踪渲染路径
    bool NeedShadowPass = false;  // 是否需要阴影Pass
    bool NeedSkyBoxPass = false;  // 是否需要天空盒Pass
    bool NeedGammaCorrection = false;  // 是否需要伽马矫正
    bool NeedAo = false;  // 是否需要SSAO
    AATpye aaType = NOAA;

    size_t width, height;  // 分辨率
    Film img;
    ThreadPool pool;  // 线程池

    Graphic graphic;
    GlobalUniform globalU;  // 全局Uniform
    std::vector<float> ZBuffer;  // Z-Buffer
    ShadowMap ShadowMap;  // 阴影Z-Buffer
    std::unique_ptr<GBuffer> gBuffer;  // GBuffer
    Film *frontBuffer{};  // 正在显示的Buffer
    Film *backBuffer{};  // 正在绘制的Buffer
    std::vector<FloatPixel> tmpBufferF;  // 后处理需要的临时Buffer,最后结果存储到F中
    std::vector<FloatPixel> tmpBufferB;
};


#endif //UNTITLED_ENGINE_H