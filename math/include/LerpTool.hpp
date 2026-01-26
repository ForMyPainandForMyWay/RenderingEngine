//
// Created by 冬榆 on 2026/1/7.
//

#ifndef RENDERINGENGINE_LERPTOOL_H
#define RENDERINGENGINE_LERPTOOL_H

#include "V2F.hpp"

struct Pixel;


// 用于SH算法的两点线性插值
V2F lerpSH(const V2F &v1, const V2F &v2, float t);

// 数值线形填充
float lerp(const float &n1, const float &n2, const float &t);

// 普通线性插值函数
template<size_t N>
VecN<N> lerp(const VecN<N> &v0, const VecN<N> &v1, float t) {
    VecN<N> result;
    for (size_t i = 0; i < N; ++i)
        result[i] = v0[i] * (1 - t) + v1[i] * t;
    return  result;
}

// Pixel线性插值,用于片元着色
// Pixel lerp(const Pixel& p1, const Pixel& p2, float t);
FloatPixel lerp(const FloatPixel& p1, const FloatPixel& p2, float t);

// 非线性插值，考虑透视校正
// 两点线性插值(用于光栅化生成vi，这时候理论上已经用不到了)
V2F lerpNoLinear(const V2F &v1, const V2F &v2, float t);

#endif //RENDERINGENGINE_LERPTOOL_H