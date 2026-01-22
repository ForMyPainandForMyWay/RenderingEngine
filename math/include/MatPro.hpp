//
// Created by 冬榆 on 2026/1/21.
//

#ifndef RENDERINGENGINE_MATVECPRO_HPP
#define RENDERINGENGINE_MATVECPRO_HPP

#include "Mat.hpp"
#include "VecPro.hpp"

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include "sse2neon.h" // 使用转换层
#endif

using Mat4 = MatMN<4,4>;
using Mat3 = MatMN<3,3>;

// ------------------------------------------------------------------
// 特化1：Mat4 * Mat4
// 编译器会优先选择这个非模板函数的重载版本
// ------------------------------------------------------------------
inline Mat4 operator * (const Mat4 &lhs, const Mat4 &rhs) {
    Mat4 result;

    // 获取数据指针
    const float* aPtr = lhs.ptr();
    const float* bPtr = rhs.ptr();
    float* rPtr = result.ptr();

    // 强制转换为 SSE 类型 (__m128)
    // 这里的逻辑是：C = A * B
    // C_row_i = sum( A_row_i_x * B_row_0, A_row_i_y * B_row_1, ... )

    // 加载 B 矩阵的 4 行
    const __m128 bLine0 = _mm_load_ps(bPtr + 0);
    const __m128 bLine1 = _mm_load_ps(bPtr + 4);
    const __m128 bLine2 = _mm_load_ps(bPtr + 8);
    const __m128 bLine3 = _mm_load_ps(bPtr + 12);

    for (int i = 0; i < 4; i++) {
        // 加载 A 的第 i 行
        // 注意：如果无法保证内存严格16字节对齐，用 _mm_loadu_ps
        __m128 aLine = _mm_load_ps(aPtr + i * 4);

        // 广播 A 的分量：
        // xxxx, yyyy, zzzz, wwww
        __m128 x = _mm_shuffle_ps(aLine, aLine, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 y = _mm_shuffle_ps(aLine, aLine, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 z = _mm_shuffle_ps(aLine, aLine, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 w = _mm_shuffle_ps(aLine, aLine, _MM_SHUFFLE(3, 3, 3, 3));

        // 乘加运算 (FMA)
        // res = x * b0 + y * b1 + z * b2 + w * b3
        __m128 sum = _mm_mul_ps(x, bLine0);
        sum = _mm_add_ps(sum, _mm_mul_ps(y, bLine1));
        sum = _mm_add_ps(sum, _mm_mul_ps(z, bLine2));
        sum = _mm_add_ps(sum, _mm_mul_ps(w, bLine3));

        // 存回结果
        _mm_store_ps(rPtr + i * 4, sum);
    }

    return result;
}

// ------------------------------------------------------------------
// 特化2：Mat4 * Vec4
// ------------------------------------------------------------------
inline Vec4 operator * (const Mat4 &mat, const Vec4 &vec) {
    // 假设 Vec4 的内存布局也是连续 float[4]
    // 且 VecN 内部有 float data[N]

    // 这里的 Vec4 是列向量处理
    // result.x = dot(row0, vec)
    // result.y = dot(row1, vec) ...

    // 加载向量 V
    const __m128 v = _mm_loadu_ps(&vec.data[0]); // Vec4 可能没对齐，用 loadu 安全

    const float* mPtr = mat.ptr();

    // 分别计算4个点积
    // 也可以用转置法优化，但这个直观写法编译器优化后效率也不错

    // Row 0
    const __m128 row0 = _mm_load_ps(mPtr + 0);
    const __m128 r0 = _mm_mul_ps(row0, v);

    // Row 1
    const __m128 row1 = _mm_load_ps(mPtr + 4);
    const __m128 r1 = _mm_mul_ps(row1, v);

    // Row 2
    const __m128 row2 = _mm_load_ps(mPtr + 8);
    const __m128 r2 = _mm_mul_ps(row2, v);

    // Row 3
    const __m128 row3 = _mm_load_ps(mPtr + 12);
    const __m128 r3 = _mm_mul_ps(row3, v);

    // 此时需要对 r0, r1, r2, r3 分别进行水平求和
    // 使用 _mm_hadd_ps (Horizontal Add)
    // t1 = (r0.x+r0.y, r0.z+r0.w, r1.x+r1.y, r1.z+r1.w)
    const __m128 t1 = _mm_hadd_ps(r0, r1);
    // t2 = (r2.x+r2.y, r2.z+r2.w, r3.x+r3.y, r3.z+r3.w)
    const __m128 t2 = _mm_hadd_ps(r2, r3);
    // sum = (t1.lo+t1.hi, t2.lo+t2.hi, ...) = (dot0, dot1, dot2, dot3)
    const __m128 sum = _mm_hadd_ps(t1, t2);

    Vec4 result;
    _mm_storeu_ps(&result.data[0], sum);
    return result;
}

// ------------------------------------------------------------------
// 特化3：矩阵加法 Mat4 + Mat4
// ------------------------------------------------------------------
inline Mat4 operator + (const Mat4 &lhs, const Mat4 &rhs) {
    Mat4 res;
    const float* a = lhs.ptr();
    const float* b = rhs.ptr();
    float* r = res.ptr();

    for(int i=0; i<4; ++i) { // 4行，每次处理一行(4个float)
        const __m128 va = _mm_load_ps(a + i*4);
        const __m128 vb = _mm_load_ps(b + i*4);
        _mm_store_ps(r + i*4, _mm_add_ps(va, vb));
    }
    return res;
}

inline Mat4 Transpose(const Mat4& m) {
    Mat4 result;

    const float* src = m.ptr();
    float* dst = result.ptr();

    // 加载 4 行
    const __m128 row0 = _mm_load_ps(src + 0);   // m00 m01 m02 m03
    const __m128 row1 = _mm_load_ps(src + 4);   // m10 m11 m12 m13
    const __m128 row2 = _mm_load_ps(src + 8);   // m20 m21 m22 m23
    const __m128 row3 = _mm_load_ps(src + 12);  // m30 m31 m32 m33

    // 第一步：拆低/高位
    const __m128 t0 = _mm_unpacklo_ps(row0, row1); // m00 m10 m01 m11
    const __m128 t1 = _mm_unpackhi_ps(row0, row1); // m02 m12 m03 m13
    const __m128 t2 = _mm_unpacklo_ps(row2, row3); // m20 m30 m21 m31
    const __m128 t3 = _mm_unpackhi_ps(row2, row3); // m22 m32 m23 m33

    // 第二步：组合成转置矩阵的行
    const __m128 col0 = _mm_movelh_ps(t0, t2); // m00 m10 m20 m30
    const __m128 col1 = _mm_movehl_ps(t2, t0); // m01 m11 m21 m31
    const __m128 col2 = _mm_movelh_ps(t1, t3); // m02 m12 m22 m32
    const __m128 col3 = _mm_movehl_ps(t3, t1); // m03 m13 m23 m33

    // 存储
    _mm_store_ps(dst + 0,  col0);
    _mm_store_ps(dst + 4,  col1);
    _mm_store_ps(dst + 8,  col2);
    _mm_store_ps(dst + 12, col3);

    return result;
}

#endif //RENDERINGENGINE_MATVECPRO_HPP