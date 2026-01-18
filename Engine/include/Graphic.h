//
// Created by 冬榆 on 2025/12/31.
//

#ifndef UNTITLED_GRAPHIC_H
#define UNTITLED_GRAPHIC_H

#include <unordered_map>
#include <vector>

#include "Shape.h"

class SkyBox;
class Shader;
class GlobalUniform;
class Engine;
class Uniform;
class RenderObjects;
struct Mesh;


// 用于绘制的类
class Graphic {
public:
    Engine *engine{};

    explicit Graphic(Engine *eg);
    void SkyPass(const SkyBox &obj, const Uniform &u, const GlobalUniform &gu, int pass=0);
    void ShadowPass(const RenderObjects &obj, const Uniform &u, const GlobalUniform &gu, int pass=1);
    void BasePass(const RenderObjects &obj, const Uniform &u, const GlobalUniform &gu, int pass=2);
    static void Clip(std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &map);
    static void ScreenMapping(std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &map, const MatMN<4, 4>&ViewPort) ;

    static void Rasterization(
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &TriMap,
        std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment>> &FragMap);  // 光栅化
    void VertexShading(
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>>& TriMap,
        const Uniform &u, const GlobalUniform &gu, const std::shared_ptr<Mesh>& mesh, int pass);  // 顶点处理
    void GeometryShading(
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>>& TriMap,
        const Uniform &u, const std::shared_ptr<Mesh>& mesh, int pass);  // 几何着色
    void FragmentShading(
        const std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment>>& fragMap,
        std::vector<F2P> &result, const Uniform &u, int pass);  // 片元着色
    static bool ZTestPix(size_t locate, float depth, std::vector<float> &ZBuffer) ;
    void Ztest(std::vector<Fragment> &TestFrag, std::vector<float> &ZBuffer) const;  // EarlyZ
    void Ztest(std::vector<F2P> &TestPix, std::vector<float> &ZBuffer) const;  // Lately-Z
    void WriteBuffer(std::vector<F2P>& f2pVec) const;

protected:
    Shader *shader;
};


#endif //UNTITLED_GRAPHIC_H