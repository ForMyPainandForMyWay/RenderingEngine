//
// Created by 冬榆 on 2025/12/31.
//

#ifndef UNTITLED_GRAPHIC_H
#define UNTITLED_GRAPHIC_H

#include <unordered_map>
#include <vector>

#include "F2P.hpp"
#include "Shape.hpp"

struct HitInfo;
struct Ray;
struct GBufferData;
enum class SSAODebugMode;
class SkyBox;
class Shader;
class GlobalUniform;
class Engine;
struct Uniform;
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
    // 几何着色阶段
    void GeometryShading(Triangle &triangle, const std::shared_ptr<Material>& material, const Uniform &u, int pass) const;

    // 处理单个三角形：完成顶点着色、裁剪、光栅化、片段着色等完整流程
    void ProcessTriangle(
        size_t idx,
        const std::vector<uint32_t>& EBOcache,
        const std::vector<Vertex>& VBO,
        const std::shared_ptr<Material>& material,
        const Uniform& u,
        const GlobalUniform& gu,
        int pass,
        const Mat4& viewPort,
        bool shadowPass = false) const;

    bool Ztest(Fragment &frag, std::vector<float> &ZBuffer) const;
    static bool ZTestPix(size_t locate, float depth, std::vector<float> &ZBuffer) ;
    // 批量写入帧缓冲区
    void WriteBuffer(const std::vector<F2P>& f2pVec) const;
    // 单个像素写入帧缓冲区
    void WriteBuffer(const F2P& f2p) const;
    void WriteGBuffer(const Fragment& frag) const;

    void RT(uint8_t SSP = 1, uint8_t maxDepth = 8) const;  // 光线追踪

    void SSAO(
        const std::vector<FloatPixel> &inBuffer,
        std::vector<FloatPixel> &outBuffer,
        const std::vector<GBufferData> &Gdata,
        const std::vector<float> &depthBuffer,
        const Mat4 &ViewMat,
        const Mat4 &Projection,
        float radius = 0.05f,
        int sampleCount = 32) const;
    void FXAA(
        std::vector<FloatPixel>& inBuffer, std::vector<FloatPixel>& outBuffer) const;
    void FXAAQ(
        std::vector<FloatPixel>& inBuffer, std::vector<FloatPixel>& outBuffer) const;
    void FXAAC(
        std::vector<FloatPixel>& inBuffer, std::vector<FloatPixel>& outBuffer) const;
};


#endif //UNTITLED_GRAPHIC_H
