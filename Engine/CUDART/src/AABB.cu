//
// Created by yyd on 2026/2/2.
//

#include "AABB.cuh"

__device__ bool IntersectAABBGPU(const AABBGPU& aabb,
    const float3& rayOrigin,
    const float3& rayDirInv,
    const float tMaxLimit,
    float& tEntry) {
    const float tx1 = (aabb.Pmin.x - rayOrigin.x) * rayDirInv.x;
    const float tx2 = (aabb.Pmax.x - rayOrigin.x) * rayDirInv.x;
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);
    const float ty1 = (aabb.Pmin.y - rayOrigin.y) * rayDirInv.y;
    const float ty2 = (aabb.Pmax.y - rayOrigin.y) * rayDirInv.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    const float tz1 = (aabb.Pmin.z - rayOrigin.z) * rayDirInv.z;
    const float tz2 = (aabb.Pmax.z - rayOrigin.z) * rayDirInv.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));
    if (tmax < 0 || tmin > tmax || tmin > tMaxLimit) return false;
    tEntry = tmin;
    return true;
}