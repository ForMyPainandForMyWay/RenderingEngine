//
// Created by yyd on 2026/2/2.
//

#ifndef RENDERINGENGINE_MATH_CUH
#define RENDERINGENGINE_MATH_CUH
#include <cuda/std/cstdint>


// 按行优先存储的矩阵
struct Mat4GPU{
    float4 r[4];
};


#ifdef __CUDACC__

__device__ __forceinline__ float4 normalize(const float4& v) {
    const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z; // 只计算xyz分量，忽略w分量
    if (lenSq > 1e-8f) {
        const float invLen = rsqrtf(lenSq);  // 使用快速倒数平方根
        return {v.x * invLen, v.y * invLen, v.z * invLen, v.w};
    }
    return {0.0f, 0.0f, 0.0f, v.w};
}

__device__ __forceinline__ float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ float4 operator*(const Mat4GPU& m, const float4& v) {
    return {dot(m.r[0], v), dot(m.r[1], v), dot(m.r[2], v), dot(m.r[3], v)};
}

__device__ __forceinline__ float4 operator/(const float4& v, const float& s) {
    return {v.x / s, v.y / s, v.z / s, v.w / s};
}

__device__ __forceinline__ float4 operator*(const float4& v, const float& s) {
    return {v.x * s, v.y * s, v.z * s, v.w * s};
}

__device__ __forceinline__ float4 operator+(const float4& a, const float4& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__device__ __forceinline__ float4 operator-(const float4& a, const float4& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return {a.x-b.x, a.y-b.y, a.z-b.z};
}

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return {a.x+b.x, a.y+b.y, a.z+b.z};
}

__device__ __forceinline__ float operator*(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a*b;
}

__device__ __forceinline__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float3 normalize(const float3& v) {
    if (const float length = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z); length > 0.0f) {
        return {v.x / length, v.y / length, v.z / length};
    }
    return {0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ float3 operator*(const float3& v, const float& s) {
    return {v.x * s, v.y * s, v.z * s};
}

__device__ __forceinline__ float3 operator/(const float3& v, const float& s) {
    return {v.x / s, v.y / s, v.z / s};
}

__device__ __forceinline__ float2 operator*(const float2& v, const float& s) {
    return {v.x * s, v.y * s};
}

__device__ __forceinline__ float2 operator+(const float2& a, const float2& b) {
    return {a.x + b.x, a.y + b.y};
}

__device__ __forceinline__ Mat4GPU Transpose4(const Mat4GPU& m) {
    return {
        make_float4(m.r[0].x, m.r[1].x, m.r[2].x, m.r[3].x),
        make_float4(m.r[0].y, m.r[1].y, m.r[2].y, m.r[3].y),
        make_float4(m.r[0].z, m.r[1].z, m.r[2].z, m.r[3].z),
        make_float4(m.r[0].w, m.r[1].w, m.r[2].w, m.r[3].w)
    };
}

__device__ __forceinline__ float3 operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ __forceinline__ float3 Hadamard(const float3& a, const float3& b) {
    return {a.x*b.x , a.y*b.y, a.z*b.z};
}

__device__ __forceinline__ float3 operator/=(float3& a, const float& b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}

__device__ __forceinline__  float GetRandomFloatGPU() {
    uint32_t x =
            (blockIdx.x * blockDim.x + threadIdx.x) * 9781u
            ^ static_cast<uint32_t>(clock64());
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    return (x >> 8) * (1.0f / 16777216.0f); // [0,1)
}

__device__ float3 Sample(float2 uv, int KdMapId);
__device__ float4 SampleCosineHemisphere(const float4& N_);
#endif

#endif //RENDERINGENGINE_MATH_CUH