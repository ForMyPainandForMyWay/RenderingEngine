//
// Created by 冬榆 on 2025/12/31.
//

#include <iostream>
#include <ranges>

#include "Engine.h"
#include "F2P.h"
#include "Mesh.h"
#include "RenderObjects.h"


Graphic::Graphic(Engine *eg) {
    this->engine = eg;
    this->shader = nullptr;
}

// 天空盒Pass
void Graphic::SkyPass(const SkyBox &obj,const Uniform &u, const GlobalUniform &gu, const int pass) {
    const auto mesh = obj.getMesh();
    std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment> > FragMap;
    {
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle> > TriMap;
        VertexShading(TriMap, u, gu, mesh, pass);
        Clip(TriMap);
        ScreenMapping(TriMap, gu.getShadowViewPort()); // 注意是ShadowViewport,用的是Light的视窗参数
        Rasterization(TriMap, FragMap);
    }
    size_t count = 0;
    for (auto &Frag: FragMap | std::views::values) {
        count += Frag.size();
    }
    std::vector<F2P> result;
    result.reserve(count);
    FragmentShading(FragMap, result, u, pass);
    WriteBuffer(result);
}

// 阴影Pass
void Graphic::ShadowPass(const RenderObjects &obj,const Uniform &u, const GlobalUniform &gu, const int pass) {
    const auto mesh = obj.getMesh();
    if (mesh == nullptr) return;
    if (mesh->getVBONums() == 0) return;
    std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment> > FragMap;
    {
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle> > TriMap;
        VertexShading(TriMap, u, gu, mesh, pass);
        Clip(TriMap);
        ScreenMapping(TriMap, gu.getShadowViewPort()); // 注意是ShadowViewport,用的是Light的视窗参数
        // 在光栅化阶段直接进行ZTest
        Rasterization(TriMap, FragMap);
    }
    size_t count = 0;
    for (auto &FragVec: FragMap | std::views::values) {
        Ztest(FragVec, engine->ShadowMap.ZBufferShadow);
        count += FragVec.size();
    }
    engine->ShadowMap.save();
}

// 基础纹理绘制,pass表示绘制层级
void Graphic::BasePass(const RenderObjects &obj,const Uniform &u, const GlobalUniform &gu, const int pass) {
    const auto mesh = obj.getMesh();
    if (mesh == nullptr) return;
    if (mesh->getVBONums() == 0) return;
    // 顶点着色阶段
    // 更新：使用脏标记+Vector更好，不过需要注意
    // 在剔除比例较高时，考虑剔除时直接新建一个vector然后逐个将有效面移动过去
    // 这涉及到CPU的分支预测，后期可以进行优化
    std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment>> FragMap;
    {
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> TriMap;
        VertexShading(TriMap, u, gu, mesh, pass);
        // 完成顶点处理阶段后进行剔除、裁剪,最后齐次除法、面剔除
        Clip(TriMap);
        // 退化检测、视口变换
        ScreenMapping(TriMap, gu.getScreenViewPort());
        // 几何着色
        GeometryShading(TriMap, u, mesh, pass);
        // 光栅化阶段，生成片元
        Rasterization(TriMap, FragMap);
    }
    // 片段着色阶段，计算每个片元的颜色、光照和阴影处理
    // Early-Z,这里不清空ZBuffer，ZBuffer在每一帧的开始清空,由Engine控制
    size_t count = 0;
    for (auto &Frag: FragMap | std::views::values) {
        Ztest(Frag, engine->ZBuffer);
        count += Frag.size();
    }
    std::vector<F2P> result;
    result.reserve(count);
    // 基础颜色/纹理贴图采样(texture自动完成各向异性过滤和MipMap)
    FragmentShading(FragMap, result, u, pass);
    for (auto &Frag: FragMap | std::views::values) {
        Ztest(Frag, engine->ZBuffer);
    }
    // 写入Buffer
    WriteBuffer(result);
}

void Graphic::WriteBuffer(const std::vector<F2P>& f2pVec) const {
    for (const auto& f2p : f2pVec) {
        if (!f2p.alive) continue;
        engine->backBuffer->WritePixle(f2p);
    }
}