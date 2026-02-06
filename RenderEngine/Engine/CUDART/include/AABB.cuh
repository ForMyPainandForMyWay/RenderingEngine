//
// Created by yyd on 2026/2/2.
//

#ifndef RENDERINGENGINE_AABB_CUH
#define RENDERINGENGINE_AABB_CUH

#include <vector_types.h>

struct AABBGPU {
    float3 Pmin;
    float3 Pmax;
};

__device__ bool IntersectAABBGPU(const AABBGPU& aabb, const float3& rayOrigin, const float3& rayDirInv, float tMaxLimit, float& tEntry);

#endif //RENDERINGENGINE_AABB_CUH