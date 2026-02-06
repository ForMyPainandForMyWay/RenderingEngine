//
// Created by 冬榆 on 2026/1/28.
//

#include "AABB.hpp"
#include <limits>
#include <algorithm>

AABB::AABB() {
    constexpr float inf = std::numeric_limits<float>::infinity();
    Pmax = Vec3{-inf, -inf, -inf}; // 初始化为无效反向包围盒
    Pmin = Vec3{inf, inf, inf};
}

bool AABB::isValid() const {
    return Pmin[0] <= Pmax[0] && Pmin[1] <= Pmax[1] && Pmin[2] <= Pmax[2];
}

void AABB::grow(const Vec3& p) {
    Pmin[0] = std::fmin(Pmin[0], p[0]); Pmin[1] = std::fmin(Pmin[1], p[1]); Pmin[2] = std::fmin(Pmin[2], p[2]);
    Pmax[0] = std::fmax(Pmax[0], p[0]); Pmax[1] = std::fmax(Pmax[1], p[1]); Pmax[2] = std::fmax(Pmax[2], p[2]);
}

void AABB::grow(const AABB& b) {
    if (!b.isValid()) return; // 忽略无效包围盒
    Pmin[0] = std::fmin(Pmin[0], b.Pmin[0]); Pmin[1] = std::fmin(Pmin[1], b.Pmin[1]); Pmin[2] = std::fmin(Pmin[2], b.Pmin[2]);
    Pmax[0] = std::fmax(Pmax[0], b.Pmax[0]); Pmax[1] = std::fmax(Pmax[1], b.Pmax[1]); Pmax[2] = std::fmax(Pmax[2], b.Pmax[2]);
}

Vec3 AABB::center() const {
    return (Pmin + Pmax) * 0.5f;
}

Vec3 AABB::extent() const {
    return Pmax - Pmin;
}

int AABB::maxDimension() const {
    Vec3 ext = extent();
    if (ext[0] > ext[1] && ext[0] > ext[2]) return 0;
    return ext[1] > ext[2] ? 1 : 2;
}

bool IntersectAABB(const AABB& aabb, const Vec3& rayOrigin, const Vec3& rayDirInv, float tMaxLimit, float& tEntry) {
    const float tx1 = (aabb.Pmin[0] - rayOrigin[0]) * rayDirInv[0];
    const float tx2 = (aabb.Pmax[0] - rayOrigin[0]) * rayDirInv[0];
    float tmin = std::min(tx1, tx2);
    float tmax = std::max(tx1, tx2);
    const float ty1 = (aabb.Pmin[1] - rayOrigin[1]) * rayDirInv[1];
    const float ty2 = (aabb.Pmax[1] - rayOrigin[1]) * rayDirInv[1];
    tmin = std::max(tmin, std::min(ty1, ty2));
    tmax = std::min(tmax, std::max(ty1, ty2));
    const float tz1 = (aabb.Pmin[2] - rayOrigin[2]) * rayDirInv[2];
    const float tz2 = (aabb.Pmax[2] - rayOrigin[2]) * rayDirInv[2];
    tmin = std::max(tmin, std::min(tz1, tz2));
    tmax = std::min(tmax, std::max(tz1, tz2));

    /* 
     * tmax < 0: AABB 在射线背面
     * tmin > tmax: 射线未穿过 AABB
     * tmin > tMaxLimit: 交点比当前找到的最近交点还远，没必要继续
    */
    if (tmax < 0 || tmin > tmax || tmin > tMaxLimit) {
        return false;
    }
    tEntry = tmin;
    return true;
}

AABB TransformAABB(const AABB& localBox, const Mat4& mat) {
    // 获取 Local AABB 的 8 个角点
    Vec3 corners[8];
    corners[0] = {localBox.Pmin[0], localBox.Pmin[1], localBox.Pmin[2]};
    corners[1] = {localBox.Pmax[0], localBox.Pmin[1], localBox.Pmin[2]};
    corners[2] = {localBox.Pmin[0], localBox.Pmax[1], localBox.Pmin[2]};
    corners[3] = {localBox.Pmax[0], localBox.Pmax[1], localBox.Pmin[2]};
    corners[4] = {localBox.Pmin[0], localBox.Pmin[1], localBox.Pmax[2]};
    corners[5] = {localBox.Pmax[0], localBox.Pmin[1], localBox.Pmax[2]};
    corners[6] = {localBox.Pmin[0], localBox.Pmax[1], localBox.Pmax[2]};
    corners[7] = {localBox.Pmax[0], localBox.Pmax[1], localBox.Pmax[2]};

    AABB worldBox; // 默认初始化为无穷大
    for (auto & corner : corners) {
        Vec4 p = {corner[0], corner[1], corner[2], 1.0f};
        Vec4 worldP = mat * p;
        // 归一化透视除法
        Vec3 wp = {worldP[0] / worldP[3], worldP[1] / worldP[3], worldP[2] / worldP[3]};
        worldBox.grow(wp);
    }
    return worldBox;
}