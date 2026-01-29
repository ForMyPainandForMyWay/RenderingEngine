//
// Created by 冬榆 on 2025/12/31.
//

#ifndef UNTITLED_GRAPHIC_H
#define UNTITLED_GRAPHIC_H

#include <unordered_map>
#include <vector>

#include "Shape.hpp"

struct HitInfo;
struct Ray;
struct GBufferData;
enum class SSAODebugMode;
class SkyBox;
class Shader;
class GlobalUniform;
class Engine;
class Uniform;
class RenderObjects;
class Mesh;


// 用于绘制的类
class Graphic {
public:
    Engine *engine{};

    explicit Graphic(Engine *eg);
    void SkyPass(const SkyBox &obj, const Uniform &u, const GlobalUniform &gu, int pass=0) const;
    void ShadowPass(const RenderObjects &obj, const Uniform &u, const GlobalUniform &gu, int pass=1) const;
    void BasePass(const RenderObjects &obj, const Uniform &u, const GlobalUniform &gu, int pass=2) const;
    void Clip(std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &map) const;
    void ScreenMapping(std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &map, const Mat4&ViewPort) const ;

    void Rasterization(
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &TriMap,
        std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment>> &FragMap) const;  // 光栅化
    void VertexShading(
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>>& TriMap,
        const Uniform &u, const GlobalUniform &gu, const std::shared_ptr<Mesh>& mesh, int pass) const;  // 顶点处理
    void GeometryShading(
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>>& TriMap,
        const Uniform &u, const std::shared_ptr<Mesh>& mesh, int pass) const;  // 几何着色
    void FragmentShading(
        const std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment>>& fragMap,
        std::vector<F2P> &result, const Uniform &u, int pass) const;  // 片元着色
    static bool ZTestPix(size_t locate, float depth, std::vector<float> &ZBuffer) ;
    void Ztest(std::vector<Fragment> &TestFrag, std::vector<float> &ZBuffer) const;  // EarlyZ
    void WriteBuffer(const std::vector<F2P>& f2pVec) const;
    void WriteGBuffer(const std::vector<Fragment>& f2pVec) const;

    void RT(uint8_t SPP = 1, uint8_t maxDepth = 8) const;  // 光线追踪

    void SSAO(
        const std::vector<FloatPixel> &inBuffer,
        std::vector<FloatPixel> &outBuffer,
        const std::vector<GBufferData> &Gdata,
        const std::vector<float> &depthBuffer,
        const Mat4 &ViewMat,
        const Mat4 &Projection,
        float radius = 0.1f,
        int sampleCount = 32) const;
    void FXAA(
        std::vector<FloatPixel>& inBuffer, std::vector<FloatPixel>& outBuffer) const;
    void FXAAQ(
        std::vector<FloatPixel>& inBuffer, std::vector<FloatPixel>& outBuffer) const;
    void FXAAC(
        std::vector<FloatPixel>& inBuffer, std::vector<FloatPixel>& outBuffer) const;
};


#endif //UNTITLED_GRAPHIC_H