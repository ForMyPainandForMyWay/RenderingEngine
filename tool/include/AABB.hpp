//
// Created by 冬榆 on 2026/1/28.
//

#ifndef RENDERINGENGINE_AABB_HPP
#define RENDERINGENGINE_AABB_HPP
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)
#include "MatPro.hpp"
#else
#include "Mat.hpp"
#endif


struct AABB {
    Vec3 Pmin;
    Vec3 Pmax;

    AABB();
    [[nodiscard]] bool isValid() const;
    void grow(const Vec3& p);  // 扩展包围盒以包含一个点
    void grow(const AABB& b);  // 扩展包围盒以包含另一个包围盒
    [[nodiscard]] Vec3 center() const;  // 获取中心点
    [[nodiscard]] Vec3 extent() const;  // 获取尺寸向量
    [[nodiscard]] int maxDimension() const;  // 获取最长轴的索引 (0:x, 1:y, 2:z)
};

// 射线与 AABB 求交
bool IntersectAABB(const AABB& aabb, const Vec3& rayOrigin, const Vec3& rayDirInv, float tMaxLimit, float& tEntry);

// 变换 AABB (将 Local AABB 的 8 个角点变换后重新生成 World AABB)
AABB TransformAABB(const AABB& localBox, const Mat4& mat);

#endif //RENDERINGENGINE_AABB_HPP