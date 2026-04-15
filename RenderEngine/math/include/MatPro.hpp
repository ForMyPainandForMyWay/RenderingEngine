//
// Created by 冬榆 on 2026/1/21.
//

#ifndef RENDERINGENGINE_MATVECPRO_HPP
#define RENDERINGENGINE_MATVECPRO_HPP

#include "Mat.hpp"
#include "VecPro.hpp"
#if ENABLE_SIMD

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include "sse2neon.h" // 使用转换层
#endif


// Mat4 * Mat4
// C = A * B
// C_{i} = A_{i}{0} * B_{0} + A_{i}{1} * B_{1} + ... + A_{i}{k} * B_{k}
inline Mat4 operator * (const Mat4 &lhs, const Mat4 &rhs) {
    Mat4 result;
    // 获取数据指针
    const float* aPtr = lhs.ptr();
    const float* bPtr = rhs.ptr();
    float* rPtr = result.ptr();

    // 加载 B 矩阵的 4 行
    const __m128 bLine0 = _mm_load_ps(bPtr + 0);
    const __m128 bLine1 = _mm_load_ps(bPtr + 4);
    const __m128 bLine2 = _mm_load_ps(bPtr + 8);
    const __m128 bLine3 = _mm_load_ps(bPtr + 12);

    for (int i = 0; i < 4; i++) {
        __m128 aLine = _mm_load_ps(aPtr + i * 4);
        // shuffle重排 A 的分量以广播计算，不涉及内存访问
        // xxxx, yyyy, zzzz, wwww
        __m128 x = _mm_shuffle_ps(aLine, aLine, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 y = _mm_shuffle_ps(aLine, aLine, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 z = _mm_shuffle_ps(aLine, aLine, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 w = _mm_shuffle_ps(aLine, aLine, _MM_SHUFFLE(3, 3, 3, 3));

        // res = x * b0 + y * b1 + z * b2 + w * b3
        __m128 sum = _mm_mul_ps(x, bLine0);
        sum = _mm_add_ps(sum, _mm_mul_ps(y, bLine1));
        sum = _mm_add_ps(sum, _mm_mul_ps(z, bLine2));
        sum = _mm_add_ps(sum, _mm_mul_ps(w, bLine3));
        _mm_store_ps(rPtr + i * 4, sum);
    }
    return result;
}

// Mat4 * Vec4
inline Vec4 operator * (const Mat4 &mat, const Vec4 &vec) {
    // Vec4 是列向量
    // result = [dot(row0, vec), dot(row1, vec), ...]
    const __m128 v = _mm_load_ps(&vec.data[0]);
    const float* mPtr = mat.ptr();

    const __m128 m0 = _mm_mul_ps(_mm_load_ps(mPtr), v);
    const __m128 m1 = _mm_mul_ps(_mm_load_ps(mPtr + 4), v);
    const __m128 m2 = _mm_mul_ps(_mm_load_ps(mPtr + 8), v);
    const __m128 m3 = _mm_mul_ps(_mm_load_ps(mPtr + 12), v);

    const __m128 t0 = _mm_hadd_ps(m0, m1);
    const __m128 t1 = _mm_hadd_ps(m2, m3);
    const __m128 result_vec = _mm_hadd_ps(t0, t1);

    Vec4 result;
    _mm_store_ps(&result.data[0], result_vec);
    return result;
}

// 矩阵加法 Mat4 + Mat4
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

// 矩阵转置
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

// 对角阵求逆
inline Mat4 diagMatInverse(const Mat4& mat) {
    // 提取对角线元素到一个 __m128 寄存器
    const __m128 diag = _mm_set_ps(
        mat[3][3], // 注意：_mm_set_ps 是 [a,b,c,d] -> [d,c,b,a] 存储顺序
        mat[2][2],
        mat[1][1],
        mat[0][0]
    );

    // 计算倒数：_mm_rcp_ps（快速近似）或精确除法
    // 若需高精度，用_mm_div_ps(1.0f, diag)
    const __m128 ones = _mm_set1_ps(1.0f);
    const __m128 inv_diag = _mm_div_ps(ones, diag); // 精确倒数

    // 将结果写回 Mat4
    alignas(16) float result_arr[4];
    _mm_store_ps(result_arr, inv_diag);

    MatMN<4, 4> result{};
    result[0][0] = result_arr[0];
    result[1][1] = result_arr[1];
    result[2][2] = result_arr[2];
    result[3][3] = result_arr[3];

    return result;
}
#endif
#endif //RENDERINGENGINE_MATVECPRO_HPP