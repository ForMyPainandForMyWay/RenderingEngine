//
// Created by 冬榆 on 2026/2/16.
//

#ifndef RENDERINGENGINE_IENGINE_H
#define RENDERINGENGINE_IENGINE_H


#if defined(_WIN32)
    #ifdef ENGINE_EXPORTS
        #define ENGINE_API __declspec(dllexport)
    #else
        #define ENGINE_API __declspec(dllimport)
    #endif
#else
    // Linux/Mac 环境
    #define ENGINE_API __attribute__((visibility("default")))
#endif

#include <vector>
#include <string>

// 前置声明，避免包含实现细节
class Lights;
enum AATpye{NOAA, FXAA, FXAAC, FXAAQ};  // 抗锯齿效果
// ID体系:0号为Camera,1号为主光源,2-4号为逐片元光源
enum sysID : size_t{CameraID, MainLightID, PixL1, PixL2, PixL3, RenderObject, VexLight, Error};
enum TfType { TRANSLATE, ROTATE, SCALE };

class IEngine {
public:
    IEngine() = default;
    virtual ~IEngine() = default;
    virtual void SetMainLight() = 0;
    virtual void SetEnvLight(uint8_t r, uint8_t g, uint8_t b, float I) = 0;
    virtual void CloseShadow() = 0;
    virtual void OpenShadow() = 0;
    virtual void CloseSky() = 0;
    virtual void OpenSky() = 0;
    virtual void SetAA(AATpye aatype) = 0;
    virtual void CloseAO() = 0;
    virtual void OpenAO() = 0;
    virtual void addTfCommand(size_t objId, sysID typeId, TfType Ttype, std::array<float, 3> value) = 0;
    virtual std::vector<std::string> addMesh(const std::string &filename) = 0;
    virtual size_t addObjects(const std::string &meshName) = 0;
    virtual sysID addPixLight(Lights &light) = 0;
    virtual size_t addVexLight(Lights &light) = 0;
    virtual void setResolution(size_t w, size_t h) = 0;

    virtual std::array<size_t, 2> getTriVexNums() = 0;

    virtual void RenderFrame(const std::vector<uint16_t>& models) = 0;  // 绘制每帧的入口，帧绘制管理
};

// 导出工厂函数，用于创建和销毁实例
extern "C" ENGINE_API IEngine *CreateEngine(size_t w, size_t h, bool Gamma, bool RT);
extern "C" ENGINE_API void DestroyEngine(const IEngine* engine);

#endif //RENDERINGENGINE_IENGINE_H