//
// Created by yyd on 2026/2/3.
//

#ifndef RENDERINGENGINE_RAY_CUH
#define RENDERINGENGINE_RAY_CUH

#include <vector_types.h>

#include "Math.cuh"

struct Ray {
    float4 orignPosi;
    float4 Direction;
};

__device__ __forceinline__ Ray TransformRayToModel(const Ray& worldRay, const Mat4GPU& invMat) {
    return {invMat * worldRay.orignPosi, invMat * worldRay.Direction};
}
#endif //RENDERINGENGINE_RAY_CUH