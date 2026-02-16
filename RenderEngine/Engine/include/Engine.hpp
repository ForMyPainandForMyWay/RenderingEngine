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
    AATpye aaType = NOAA;
};

class Engine: public IEngine {
public:
    Engine(size_t w, size_t h, bool Gamma=false, bool RT=false);
    ~Engine() override;
    void SetMainLight() override;
    void SetEnvLight(uint8_t r, uint8_t g, uint8_t b, float I) override;

    void CloseShadow() override { settings[!renderSetting].NeedShadowPass = false; }
    void OpenShadow() override { settings[!renderSetting].NeedShadowPass = true; }
    void CloseSky() override { settings[!renderSetting].NeedSkyBoxPass = false; }
    void OpenSky() override { settings[!renderSetting].NeedSkyBoxPass = true; }
    void SetAA(const AATpye aatype) override { settings[!renderSetting].aaType = aatype; }
    void CloseAO() override { settings[!renderSetting].NeedAo = false; }
    void OpenAO() override { settings[!renderSetting].NeedAo = true; }
    void addTfCommand(const TfCmd &cmd);
    void addTfCommand(size_t objId, sysID typeId, TfType Ttype, std::array<float, 3> value) override;
    std::vector<std::string> addMesh(const std::string &filename) override;
    size_t addObjects(const std::string &meshName) override;
    sysID addPixLight(Lights &light) override;
    size_t addVexLight(Lights &light) override;
    void setResolution(size_t w, size_t h) override;
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
    SettingCache settings[2];  // 设置参数双缓冲
    std::atomic<bool> renderSetting;  // 双缓冲指针，指示渲染流程中用到的配置

    Film img;
    ThreadPool pool;  // 线程池

    Graphic graphic;
    GlobalUniform globalU;  // 全局Uniform
    std::vector<float> ZBuffer;  // Z-Buffer
    std::shared_ptr<ShadowMap> SdMap;  // 阴影Z-Buffer
    std::unique_ptr<GBuffer> gBuffer;  // GBuffer
    Film *frontBuffer{};  // 正在显示的Buffer
    Film *backBuffer{};  // 正在绘制的Buffer
    std::vector<FloatPixel> tmpBufferF;  // 后处理需要的临时Buffer,最后结果存储到F中
    std::vector<FloatPixel> tmpBufferB;
};


#endif //UNTITLED_ENGINE_H