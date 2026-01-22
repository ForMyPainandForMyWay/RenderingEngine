//
// Created by 冬榆 on 2026/1/21.
//

#ifndef RENDERINGENGINE_VECPRO_HPP
#define RENDERINGENGINE_VECPRO_HPP
#include "Vec.hpp"

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include "sse2neon.h" // 使用转换层
#endif

using Vec3 = VecN<3>;
using Vec4 = VecN<4>;

// Vec4 加法
inline Vec4 operator+(const Vec4& lhs, const Vec4& rhs) {
    Vec4 res;
    _mm_store_ps(res.ptr(), _mm_add_ps(_mm_load_ps(lhs.ptr()), _mm_load_ps(rhs.ptr())));
    return res;
}

// Vec4 减法
inline Vec4 operator-(const Vec4& lhs, const Vec4& rhs) {
    Vec4 res;
    _mm_store_ps(res.ptr(), _mm_sub_ps(_mm_load_ps(lhs.ptr()), _mm_load_ps(rhs.ptr())));
    return res;
}

// Vec4 * float (标量乘法)
inline Vec4 operator*(const Vec4& lhs, const float scalar) {
    Vec4 res;
    const __m128 s = _mm_set1_ps(scalar); // 将 scalar 广播到 4 个分量
    _mm_store_ps(res.ptr(), _mm_mul_ps(_mm_load_ps(lhs.ptr()), s));
    return res;
}

// Vec4 点积 (Dot Product)
inline float dot(const Vec4& lhs, const Vec4& rhs) {
    // _mm_dp_ps (SSE4.1)
    // 0xF1 参数解释:
    // 高4位 (0xF=1111): 计算所有4个分量的乘积之和
    // 低4位 (0x1=0001): 将结果存放在返回寄存器的最低位
    const __m128 dot = _mm_dp_ps(_mm_load_ps(lhs.ptr()), _mm_load_ps(rhs.ptr()), 0xF1);
    return _mm_cvtss_f32(dot); // 提取最低位的 float
}

// Vec4 模长 (Length)
inline float length(const Vec4& v) {
    // Length = sqrt(Dot(v, v))
    const __m128 dot = _mm_dp_ps(_mm_load_ps(v.ptr()), _mm_load_ps(v.ptr()), 0xF1);
    return _mm_cvtss_f32(_mm_sqrt_ss(dot));
}

// Vec4 归一化 (Normalize)
inline Vec4 normalize(const Vec4& v) {
    Vec4 res;
    const __m128 vec = _mm_load_ps(v.ptr());
    // 计算点积，结果广播到所有分量 (0xFF)
    const __m128 dot = _mm_dp_ps(vec, vec, 0xFF);
    // rsqrt 计算 1/sqrt(dot) 的近似值 (速度极快)
    // 如果需要高精度，改用 _mm_div_ps(vec, _mm_sqrt_ps(dot))
    const __m128 invLen = _mm_rsqrt_ps(dot);
    _mm_store_ps(res.ptr(), _mm_mul_ps(vec, invLen));
    return res;
}

inline VecN<4> Hadamard(const VecN<4> &a, const VecN<4> &b) {
    VecN<4> result;

    // 假设 VecN<4> 内部连续存储 float[4] 并16字节对齐
    __m128 va = _mm_load_ps(&a.data[0]);
    __m128 vb = _mm_load_ps(&b.data[0]);
    __m128 vr = _mm_mul_ps(va, vb);
    _mm_store_ps(&result.data[0], vr);

    return result;
}


#endif //RENDERINGENGINE_VECPRO_HPP