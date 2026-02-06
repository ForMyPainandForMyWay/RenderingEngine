//
// Created by yyd on 2026/2/2.
//

#include "TransfomTool.cuh"
#include "AABB.cuh"
#include "AABB.hpp"
#include "Mat.hpp"
#include "Math.cuh"

float4 Vec4ToFloat4(const Vec4& vec4) {
    return {vec4[0], vec4[1], vec4[2], vec4[3]};
}

float3 Vec3ToFloat3(const Vec3& vec3) {
    return {vec3[0], vec3[1], vec3[2]};
}

float2 Vec2ToFloat2(const VecN<2>& vec2) {
    return {vec2[0], vec2[1]};
}

Mat4GPU Mas4ToGPU(const Mat4& mat4) {
    return {Vec4ToFloat4(mat4[0]), Vec4ToFloat4(mat4[1]), Vec4ToFloat4(mat4[2]), Vec4ToFloat4(mat4[3])};
}

AABBGPU AABBToGPU(const AABB& aabb) {
    return {Vec3ToFloat3(aabb.Pmin), Vec3ToFloat3(aabb.Pmax)};
}
