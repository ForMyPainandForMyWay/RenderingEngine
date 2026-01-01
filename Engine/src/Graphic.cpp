//
// Created by 冬榆 on 2025/12/31.
//
#include <ranges>

#include "Graphic.h"
#include "MathTool.hpp"
#include "Mesh.h"
#include "RenderObjects.h"
#include "Shader.h"


Graphic::Graphic(Engine *eg, FrameBuffer *buffer) {
    this->engine = eg;
    this->renderBuffer = buffer;
    this->shader = nullptr;
}

// 绘制模型,pass表示绘制层级
void Graphic::DrawModel(const RenderObjects &obj,
                        const Uniform &u,
                        const int pass) {
    const auto mesh = obj.getMesh();
    if (mesh == nullptr) return;
    if (mesh->getVBONums() == 0) return;

    // 顶点着色阶段
    // 更新：使用脏标记+Vector更好，不过需要注意
    // 在剔除比例较高时，考虑剔除时直接新建一个vector然后逐个将有效面移动过去
    // 这涉及到CPU的分支预测，后期可以进行优化
    std::unordered_map<Material*, std::vector<Triangle>> map;
    for (const auto &sub : *mesh) {
        shader = sub.getMaterial()->getShader(pass);
        Material* material = sub.getMaterial();
        const auto oft = sub.getOffset();
        const auto oftEnd = sub.getIdxCount() + oft;
        map[material].reserve((oftEnd-oft) / 3);  // 预分配内存
        for (auto idx = oft; idx < oftEnd; idx+=3) {
            V2F v1 = VertexShading(mesh->VBO[mesh->EBO[idx]]);
            V2F v2 = VertexShading(mesh->VBO[mesh->EBO[idx+1]]);
            V2F v3 = VertexShading(mesh->VBO[mesh->EBO[idx+2]]);
            map[material].emplace_back(v1, v2, v3);
        }
    }

    // 完成顶点处理阶段后进行剔除、裁剪,最后齐次除法、面剔除
    Clip(map);

    // 视口变换

    // 光栅化

}

// 顶点着色后处理
void Graphic::Clip(std::unordered_map<Material*, std::vector<Triangle>> &map) {
    for (auto& triangles: map | std::views::values) {
        // 三点组成一个三角形
        std::vector<Triangle> result;
        for (auto& triangle : triangles) {
            V2F &p1 = triangle[0];
            V2F &p2 = triangle[1];
            V2F &p3 = triangle[2];

            // 快速剔除全部在外的
            if (AllVertexOutside(p1, p2, p3)) {
                triangle.alive = false;
                continue;
            }
            std::vector<Triangle> tri = {triangle};
            // 裁剪
            const bool clip = ! AllVertexInside(p1, p2, p3);
            if (clip) {
                triangle.alive = false;  // 先剔除原来的旧三角
                tri = PolyClip(p1, p2, p3);  // S-H算法.直接返回切分后的三角数组
            }
            for (auto &t : tri) {
                PersDiv(t);  // 透视除法
                FaceClip(t);  // 面剔除
                if (clip) result.emplace_back(t);
            }
        }
        triangles.insert(triangles.end(),
                result.begin(), result.end());
    }
}


V2F Graphic::VertexShading(const Vertex &vex) const {
    return shader->VertexShader(vex);
}

