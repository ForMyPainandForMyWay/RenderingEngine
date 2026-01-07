//
// Created by 冬榆 on 2025/12/31.
//

#ifndef UNTITLED_GRAPHIC_H
#define UNTITLED_GRAPHIC_H

#include <unordered_map>
#include <vector>


class Engine;
// class FrameBuffer;
class Shader;
class Uniform;
class RenderObjects;
class Material;
struct Triangle;
struct V2F;
struct F2P;
struct Vertex;
struct Fragment;
struct Mesh;

// 用于绘制的类
class Graphic {
public:
    Engine *engine{};

    explicit Graphic(Engine *eg);
    void DrawModel(const RenderObjects &obj, const Uniform &u, int pass=0);
    static void Clip(std::unordered_map<Material*, std::vector<Triangle>> &map);
    void ScreenMapping(std::unordered_map<Material*, std::vector<Triangle>> &map) const;
    static void Rasterization(
        std::unordered_map<Material*, std::vector<Triangle>> &TriMap,
        std::unordered_map<Material*, std::vector<Fragment>> &FragMap);
    static std::vector<Fragment> Rasterizing(Triangle &tri);
    void VertexShading(
        std::unordered_map<Material*, std::vector<Triangle>>& TriMap,
        const Uniform &u, const Mesh *mesh, int pass);  // 顶点处理
    void FragmentShading(
        const std::unordered_map<Material*, std::vector<Fragment>>& fragMap,
        std::vector<F2P> &result, const Uniform &u, int pass);  // 片元着色
    void Ztest(std::vector<Fragment> &TestFrag) const;  // EarlyZ
    void Ztest(std::vector<F2P> &TestPix) const;  // Lately-Z
    void WriteBuffer(const std::vector<F2P>& f2pVec) const;

protected:
    Shader *shader;
};


#endif //UNTITLED_GRAPHIC_H