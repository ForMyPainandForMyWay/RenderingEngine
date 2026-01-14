//
// Created by 冬榆 on 2025/12/29.
//

#include <ranges>

#include "RenderObjects.h"
#include "ClipTool.h"
#include "Engine.h"
#include "F2P.h"
#include "MathTool.hpp"
#include "Mesh.h"
#include "RasterTool.hpp"
#include "BlinnShader.h"


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
        else if (objId == 1 && mainLight!= nullptr) applyCmd(*mainLight);  // light 提供相同接口
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
            vexList.emplace_back(BlinnShader::VertexShader(vex, u));
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

// 顶点着色后处理: 视锥剔除、SH裁剪、透视除法、背面剔除、深度映射
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
                    PersDiv(t);   // 透视除法->NDC空间
                    FaceClip(t);  // 背面剔除
                    // DepthMap(t);  // 深度映射
                    result.emplace_back(t);
                }
            } else {
                PersDiv(triangle);
                FaceClip(triangle);
                // DepthMap(triangle);
            }
        }
        triangles.insert(triangles.end(),
                result.begin(), result.end());
    }
}

// 视口变换把坐标从NDC转换到Screen、面积退化检测
void Graphic::ScreenMapping(std::unordered_map<Material *, std::vector<Triangle>> &map, const MatMN<4, 4>&ViewPort) {
    for (auto& triangles : map | std::views::values) {
        for (auto& tri : triangles) {
            if (!tri.alive) continue;
            tri[0].clipPosi = ViewPort * tri[0].clipPosi;
            tri[1].clipPosi = ViewPort * tri[1].clipPosi;
            tri[2].clipPosi = ViewPort * tri[2].clipPosi;
            DegenerateClip(tri);  // 退化检测(面积过小的三角)
        }
    }
}

// ZTest组件
bool Graphic::ZTestPix(const size_t locate, const float depth, std::vector<float> &ZBuffer) {
    if (ZBuffer[locate] > depth) {
        ZBuffer[locate] = depth;
        return true;
    }
    return false;
}

// Early-Z,传入Fragment
void Graphic::Ztest(std::vector<Fragment> &TestFrag, std::vector<float> &ZBuffer) const {
    int keptCount = 0;
    for (auto &pix : TestFrag) {
        if (const auto locate = pix.x + pix.y * engine->width;
            ZTestPix(locate, pix.depth, ZBuffer)) {
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
            result.emplace_back(
                shader->FragmentShader(
                    frag,material,
                engine->lights,
                engine->mainLight,
                engine->ShadowMap,
                engine->envLight,
                engine->globalU,
                engine->NeedShadowPass));
        }
    }
}

// Lately-Z,传入F2P,不能清除ZBuffer
void Graphic::Ztest(std::vector<F2P> &TestPix, std::vector<float> &ZBuffer) const {
    for (auto &pix : TestPix) {
        if (const auto locate = pix.x + pix.y * engine->width;
            ZBuffer[locate] > pix.depth ) {
            ZBuffer[locate] = pix.depth;
            pix.keep();
            }
        else pix.drop();  // 后续frag不再着色
    }
}

