//
// Created by yyd on 2026/2/2.
//

#ifndef RENDERINGENGINE_TRANSFOMTOOL_CUH
#define RENDERINGENGINE_TRANSFOMTOOL_CUH

#include "Mat.hpp"

struct AABB;
struct AABBGPU;
struct Mat4GPU;

float4 Vec4ToFloat4(const Vec4& vec4);
float3 Vec3ToFloat3(const Vec3& vec3);
float2 Vec2ToFloat2(const VecN<2>& vec2);
Mat4GPU Mas4ToGPU(const Mat4& mat4);

AABBGPU AABBToGPU(const AABB& aabb);
#endif //RENDERINGENGINE_TRANSFOMTOOL_CUH