//
// Created by yyd on 2026/2/2.
//

#include "Math.cuh"
#include "PathTracing.cuh"

__device__ float3 Sample(const float2 uv, const int KdMapId) {
    if (KdMapId == -1) {
        return {0.5, 0.5, 0.5};
    }
    // 拿到纹理对象
    const cudaTextureObject_t& texObj = texObjsGPU[KdMapId];
    const float4& texel =tex2D<float4>(texObj, uv.x, uv.y);
    return {texel.x, texel.y, texel.z};
}

__device__ float4 SampleCosineHemisphere(const float4& N_) {
    const float3 N = {N_.x, N_.y, N_.z};
    const float r1 = GetRandomFloatGPU();
    const float r2 = GetRandomFloatGPU();

    // 计算局部空间坐标
    // theta 根据 sqrt(1-r2) 分布，偏向法线
    const float phi = 2.0f * 3.1415926f * r1;
    const float sinTheta = sqrtf(1.0f - r2);
    const float cosTheta = sqrtf(r2);
    const float x = sinTheta * cos(phi);
    const float y = sinTheta * sin(phi);
    const float z = cosTheta;
    // 构建 TBN 矩阵(将局部坐标转到世界坐标)
    const float3 up = fabs(N.z) < 0.999f ? float3{0.0f, 0.0f, 1.0f} : float3{1.0f, 0.0f, 0.0f};
    const float3 T = normalize(cross(up, N));
    const float3 B = cross(N, T);
    const auto [Rex, Rey, Rez] = normalize(T * x + B * y + N * z);
    // 转换并返回世界空间方向(齐次)
    return {Rex, Rey, Rez, 0.0f};
}
