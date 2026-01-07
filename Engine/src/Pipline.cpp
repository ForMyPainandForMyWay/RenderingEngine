//
// Created by 冬榆 on 2025/12/29.
//

#include <ranges>

#include "ClipTool.h"
#include "Engine.h"
#include "F2P.h"
#include "MathTool.hpp"
#include "Mesh.h"
#include "RasterTool.h"
#include "Shader.h"


// 应用阶段，对实例应用变换
void Engine::Application() {
    while (!tfCommand.empty()) {
        auto [objId, type, value] = tfCommand.front();
        tfCommand.pop();
        // 泛型 lambda：接受任意具有 updateP/Q/S 接口的对象
        auto applyCmd = [&](auto& obj) {
            switch (type) {
                case TfCmd::TRANSLATE: obj.updateP(value);break;
                case TfCmd::ROTATE: obj.updateQ(Euler2Quaternion(value));break;  // 注意转弧度制
                case TfCmd::SCALE: obj.updateS(value);break;
                default: break;
            }
        };
        if (objId == 0) applyCmd(camera);         // camera 提供 updateP/Q/S
        else applyCmd(renderObjs.at(objId));  // renderObj 也提供相同接口
    }
}

// 顶点着色
void Graphic::VertexShading(
    std::unordered_map<Material*, std::vector<Triangle>>& TriMap,
    const Uniform &u, const Mesh *mesh, const int pass) {
    for (const auto &sub : *mesh) {
        shader = sub.getMaterial()->getShader(pass);
        Material* material = sub.getMaterial();
        // 计算所有顶点并缓存一份
        std::vector<V2F> vexList;
        vexList.reserve(mesh->VBO.size());
        for (const auto &vex : mesh->VBO) {
            vexList.emplace_back(Shader::VertexShader(vex, u));
        }
        const auto oft = sub.getOffset();
        const auto oftEnd = sub.getIdxCount() + oft;
        TriMap[material].reserve((oftEnd-oft-1) / 3);  // 预分配内存
        for (auto idx = oft; idx < oftEnd; idx+=3) {
            V2F v1 =  vexList[mesh->EBO[idx]];
            V2F v2 =  vexList[mesh->EBO[idx+1]];
            V2F v3 =  vexList[mesh->EBO[idx+2]];
            TriMap[material].emplace_back(v1, v2, v3);
        }
    }
}

// 顶点着色后处理: 视锥剔除、SH裁剪、
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
            std::vector<Triangle> tris;
            // 裁剪
            const bool clip = !AllVertexInside(p1, p2, p3);
            if (clip) {
                triangle.alive = false;  // 先剔除原来的旧三角
                tris = PolyClip(p1, p2, p3);  // S-H算法.直接返回切分后的三角数组
            }
            if (clip) {
                for (auto &t : tris) {
                    PersDiv(t);  // 透视除法->NDC空间
                    FaceClip(t);  // 背面剔除
                    result.emplace_back(t);
                }
            } else {
                PersDiv(triangle);
                FaceClip(triangle);
            }
        }
        triangles.insert(triangles.end(),
                result.begin(), result.end());
    }
}

// 视口变换把坐标从NDC转换到Screen、面积退化检测
void Graphic::ScreenMapping(std::unordered_map<Material *, std::vector<Triangle>> &map) const {
    for (auto& triangles : map | std::views::values) {
        for (auto& tri : triangles) {
            if (!tri.alive) continue;
            tri[0].position = engine->globalU.getViewPort() * tri[0].position;
            tri[1].position = engine->globalU.getViewPort() * tri[1].position;
            tri[2].position = engine->globalU.getViewPort() * tri[2].position;
            DegenerateClip(tri);  // 退化检测(面积过小的三角)
        }
    }
}

// 光栅化接口
void Graphic::Rasterization(
    std::unordered_map<Material*, std::vector<Triangle>> &TriMap,
    std::unordered_map<Material*, std::vector<Fragment>> &FragMap) {
    std::unordered_map<Material*, std::vector<Fragment>> fragMap;
    for (auto& [material, triangles] : TriMap) {
        std::vector<Fragment> fragVec;
        for (auto& tri : triangles) {
            if (!tri.alive) continue;  // 背面剔除、退化剔除
            // 光栅化并返回该三角形的片元序列
            std::vector<Fragment> triFrags = Rasterizing(tri);
            // 加入序列到fragVec中
            fragVec.insert(fragVec.end(), triFrags.begin(), triFrags.end());
        }
        // 设置片元序列
        FragMap.emplace(material, fragVec);
    }
}

// 光栅化
std::vector<Fragment> Graphic::Rasterizing(Triangle &tri) {
    std::vector<Fragment> result{};
    // if (!DegenerateClip(tri)) {
    //     tri.alive = false;
    //     return result;  // 退化检测(面积过小的三角)
    // }
    // 光栅化转移到视口变换时进行
    sortTriangle(tri);  // 三角形顶点排序
    ScanLine(tri, result);  // 扫描线算法光栅化
    return result;
}

// Early-Z,传入Fragment
void Graphic::Ztest(std::vector<Fragment> &TestFrag) const {
    int keptCount = 0;
    for (auto &pix : TestFrag) {
        if (const auto locate = pix.x + pix.y * engine->width;
            engine->ZBuffer[locate] > pix.depth ) {
            engine->ZBuffer[locate] = pix.depth;
            pix.keep();
            keptCount++;
            }
        else pix.drop();  // 后续frag不再着色
    }
}

// 片元着色器
void Graphic::FragmentShading(
    const std::unordered_map<Material *, std::vector<Fragment> >& fragMap,
    std::vector<F2P> &result, const Uniform &u, const int pass) {
    for (auto& [material, fragVec] : fragMap) {
        shader = material->getShader(pass);
        shader->setMaterial(material);
        for (auto& frag : fragVec) {
            if (!frag.alive) continue;
            result.emplace_back(shader->FragmentShader(frag, u));
        }
    }
}

// Lately-Z,传入F2P,不能清除ZBuffer
void Graphic::Ztest(std::vector<F2P> &TestPix) const {
    for (auto &pix : TestPix) {
        if (const auto locate = pix.x + pix.y * engine->width;
            engine->ZBuffer[locate] > pix.depth ) {
            engine->ZBuffer[locate] = pix.depth;
            pix.keep();
            }
        else pix.drop();  // 后续frag不再着色
    }
}

