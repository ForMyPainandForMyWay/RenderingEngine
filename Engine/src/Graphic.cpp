//
// Created by 冬榆 on 2025/12/31.
//

#include <ranges>
#include <iostream>

#include "Graphic.h"
#include "Engine.h"
#include "F2P.h"
#include "Mesh.h"
#include "RenderObjects.h"
#include "Shader.h"


Graphic::Graphic(Engine *eg) {
    this->engine = eg;
    this->shader = nullptr;
}

// 绘制模型,pass表示绘制层级
void Graphic::DrawModel(const RenderObjects &obj,const Uniform &u,const int pass) {
    const auto mesh = obj.getMesh();
    if (mesh == nullptr) return;
    if (mesh->getVBONums() == 0) return;
    // 顶点着色阶段
    // 更新：使用脏标记+Vector更好，不过需要注意
    // 在剔除比例较高时，考虑剔除时直接新建一个vector然后逐个将有效面移动过去
    // 这涉及到CPU的分支预测，后期可以进行优化
    std::unordered_map<Material*, std::vector<Fragment>> FragMap;
    {
        std::unordered_map<Material*, std::vector<Triangle>> TriMap;
        VertexShading(TriMap, u, mesh, pass);
        // 完成顶点处理阶段后进行剔除、裁剪,最后齐次除法、面剔除
        Clip(TriMap);
        // 视口变换
        ScreenMapping(TriMap);
        // 退化检测、光栅化
        Rasterization(TriMap, FragMap);
    }
    // 片段着色阶段，计算每个片元的颜色、光照和阴影处理
    // Early-Z,这里不清空ZBuffer，ZBuffer在每一帧的开始清空,由Engine控制
    size_t count = 0;
    for (auto &Frag : FragMap | std::views::values) {
        Ztest(Frag);
        count += Frag.size();
    }
    std::vector<F2P> f2pVec;
    f2pVec.reserve(count);
    // 基础颜色/纹理贴图采样(texture自动完成各向异性过滤和MipMap)
    // 光照计算(phong或者PBR)
    FragmentShading(FragMap, f2pVec, u, pass);
    // Lately-Z
    Ztest(f2pVec);
    // 后效处理，如空间环境光遮蔽（但是好像是由引擎在帧控制器控制的）

    // 写入帧缓冲
    WriteBuffer(f2pVec);
}

void Graphic::WriteBuffer(const std::vector<F2P>& f2pVec) const {
    for (const auto& f2p : f2pVec)
        engine->backBuffer->WritePixle(f2p);
}