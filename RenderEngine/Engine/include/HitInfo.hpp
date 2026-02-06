//
// Created by 冬榆 on 2026/1/24.
//

#ifndef RENDERINGENGINE_HITINFO_HPP
#define RENDERINGENGINE_HITINFO_HPP
#include <cstdint>
#include <memory>
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)
#include "MatPro.hpp"
#else
#include "Vec.hpp"
#include "Mat.hpp"
#endif

struct Material;

struct HitInfo {
    float t{};        // 射线距离
    Vec4 hitPos;      // 碰撞点坐标 (模型空间)
    Vec4 hitNormal;   // 碰撞点法线 (模型空间)
    VecN<2> hitUV;    // 碰撞点纹理坐标
    uint16_t model{};   // 模型ID
    std::shared_ptr<Material> mat{}; // 材质指针
    void trans2World(const Mat4& ModelMat, const Mat4& NormalWorldMat);
};


#endif //RENDERINGENGINE_HITINFO_HPP