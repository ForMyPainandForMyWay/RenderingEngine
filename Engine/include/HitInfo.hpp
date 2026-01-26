//
// Created by 冬榆 on 2026/1/24.
//

#ifndef RENDERINGENGINE_HITINFO_HPP
#define RENDERINGENGINE_HITINFO_HPP

#include "MatPro.hpp"
#include "VecPro.hpp"

struct Material;

struct HitInfo {
    const Material *material{};
    float t{};        // 射线距离
    Vec4 hitPos;      // 碰撞点坐标 (模型空间)
    Vec4 hitNormal;   // 碰撞点法线 (模型空间)
    VecN<2> hitUV;    // 碰撞点纹理坐标
    uint16_t model{};   // 模型ID
    // 如果需要，还可以加 VertexColor 等

    void trans2World(const Mat4& ModelMat, const Mat4& NormalWorldMat);
};


#endif //RENDERINGENGINE_HITINFO_HPP