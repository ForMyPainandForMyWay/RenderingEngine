//
// Created by 冬榆 on 2025/12/29.
//

#ifndef UNTITLED_ENGINE_H
#define UNTITLED_ENGINE_H

#include <queue>

#include "IEngine.hpp"
#include "BVH.hpp"
#include "Camera.hpp"
#include "GBuffer.hpp"
#include "Graphic.hpp"
#include "Uniform.hpp"
#include "Lights.hpp"
#include "ShadowMap.hpp"
#include "SkyBox.hpp"
#include "SwapChain.hpp"
#include "thread_pool.hpp"


struct BLAS;
class ThreadPool;

// 变换指令(累积变换)
struct TfCmd {
    size_t objId{};
    sysID typeId{};
    TfType type{};
    Vec3 value;  // 移动/旋转/缩放值
};

// 参数缓冲区
struct SettingCache {
    bool IsRT = false;  // 是否是光线追踪渲染路径
    bool NeedShadowPass = false;  // 是否需要阴影Pass
    bool NeedSkyBoxPass = false;  // 是否需要天空盒Pass
    bool NeedGammaCorrection = false;  // 是否需要伽马矫正
    bool NeedAo = false;  // 是否需要SSAO
    float fov = 45;
    float near = 0.1f;
    float far = 20.0f;
    AATpye aaType = NOAA;
};

class Engine: public IEngine {
public:
    Engine(size_t w, size_t h, bool Gamma=false, bool RT=false);
    ~Engine() override;
    void SetMainLight(uint8_t r, uint8_t g, uint8_t b, float I) override;
    void SetEnvLight(uint8_t r, uint8_t g, uint8_t b, float I) override;
    void SetPixLight(sysID plId, uint8_t r, uint8_t g, uint8_t b, float I) override;

    void SetRtMode() override { std::lock_guard lock(settingMtx); settings[1].IsRT = true; }
    void SetRasMode() override { std::lock_guard lock(settingMtx); settings[1].IsRT = false; }

    void CloseShadow() override { std::lock_guard lock(settingMtx); settings[1].NeedShadowPass = false; }
    void OpenShadow() override { std::lock_guard lock(settingMtx); settings[1].NeedShadowPass = true; }
    void CloseSky() override { std::lock_guard lock(settingMtx); settings[1].NeedSkyBoxPass = false; }
    void OpenSky() override { std::lock_guard lock(settingMtx); settings[1].NeedSkyBoxPass = true; }
    void SetAA(const AATpye aatype) override { std::lock_guard lock(settingMtx); settings[1].aaType = aatype; }
    void CloseAO() override { std::lock_guard lock(settingMtx); settings[1].NeedAo = false; }
    void OpenAO() override { std::lock_guard lock(settingMtx); settings[1].NeedAo = true; }
    void addTfCommand(const TfCmd &cmd);
    void addTfCommand(size_t objId, sysID typeId, TfType Ttype, std::array<float, 3> value) override;
    std::vector<std::string> addMesh(const std::string &filename) override;
    size_t addObjects(const std::string &meshName) override;
    sysID addPixLight(uint8_t r, uint8_t g, uint8_t b, LType type) override;
    size_t addVexLight(uint8_t r, uint8_t g, uint8_t b, LType type) override;
    void setResolution(size_t w, size_t h) override;

    void setCameraFov(const float fov) override { std::lock_guard lock(settingMtx); settings[1].fov = fov; }
    void setCameraNear(const float near) override { std::lock_guard lock(settingMtx); settings[1].near = near; }
    void setCameraFar(const float far) override { std::lock_guard lock(settingMtx); settings[1].far = far; }
    // 光线追踪
    void BuildTLAS(const std::vector<uint16_t>& models);
    std::optional<HitInfo> GetClosestHit(const Ray &worldRay) const;


    void Application();  // 应用物体、相机与光源的变换
    void RenderFrame(const std::vector<uint16_t>& models) override;  // 绘制每帧的入口，帧绘制管理
    void BeginFrame();   // 初始化帧
    void EndFrame();  // 交付帧
    void DrawScene(const std::vector<uint16_t>& models);  // 绘制场景
    void DrawScenceRT(const std::vector<uint16_t>& models);  // 光线追踪路径
    void PostProcess();  // 后处理
    void Write2Front();  // 写入绘制缓冲区(交付),自行转换伽马矫正

    std::array<size_t, 2> getTriVexNums() override;  // 返回 [三角形、顶点的数目]
    void startLoop(std::vector<uint16_t> objs, IFrameReceiver *receiver) override;
    void stopLoop() override;

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
    std::queue<TfCmd> tfCommand;
    bool NeedGammaCorrection = false;  // 是否需要伽马矫正
    size_t width = 400, height = 400;  // 分辨率
    SettingCache settings[2];  // 设置参数双缓冲,0号为当前使用的参数,1号为等待调整的参数
    std::mutex settingMtx;  // 设置缓冲区的锁
    std::atomic<bool> renderLoop;
    std::atomic<bool> pullLoop;

    Film img;
    ThreadPool pool;  // 线程池
    std::thread RendWorker;  // 渲染循环线程
    std::thread PullWorker;

    Graphic graphic;
    GlobalUniform globalU;  // 全局Uniform
    std::vector<float> ZBuffer;  // Z-Buffer
    std::shared_ptr<ShadowMap> SdMap;  // 阴影Z-Buffer
    std::unique_ptr<GBuffer> gBuffer;  // GBuffer
    std::shared_ptr<Film> rendBuffer;  // 需要绘制的buffer的指针
    std::unique_ptr<SwapChain> swapChain;  // 交换链
    std::vector<FloatPixel> tmpBufferF;  // 后处理需要的临时Buffer,最后结果存储到F中
    std::vector<FloatPixel> tmpBufferB;
};


#endif //UNTITLED_ENGINE_H