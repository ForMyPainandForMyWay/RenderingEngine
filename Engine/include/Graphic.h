//
// Created by 冬榆 on 2025/12/31.
//

#ifndef UNTITLED_GRAPHIC_H
#define UNTITLED_GRAPHIC_H

#include <unordered_map>
#include <vector>


class Engine;
class FrameBuffer;
class Shader;
class Uniform;
class RenderObjects;
class Material;
struct Triangle;
struct V2F;
struct Vertex;
struct Fragment;

// 用于绘制的类
class Graphic {
public:
    Engine *engine{};

    explicit Graphic(Engine *eg, FrameBuffer *buffer);
    void DrawModel(const RenderObjects &obj, const Uniform &u, int pass=0);
    static void Clip(std::unordered_map<Material*, std::vector<Triangle>> &map);
    void ScreenMapping(std::unordered_map<Material*, std::vector<Triangle>> &map) const;
    static void Rasterization(
        std::unordered_map<Material*, std::vector<Triangle>> &TriMap,
        std::unordered_map<Material*, std::vector<Fragment>> &FragMap) ;
    static std::vector<Fragment> Rasterizing(Triangle &tri) ;

    static V2F VertexShading(const Vertex &vex, const Uniform &u) ;  // 顶点处理
    void PrimitiveAssembly();  // 图元组装、裁剪
    void ClipProcess();   // 视锥剔除+顶点裁剪+透视除法
    void ScreenMapping();  // 视口变换

protected:
    FrameBuffer *renderBuffer{};  // 绘制缓冲区
    Shader *shader;
};


#endif //UNTITLED_GRAPHIC_H