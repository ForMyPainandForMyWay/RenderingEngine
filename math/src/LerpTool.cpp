//
// Created by 冬榆 on 2026/1/14.
//

#ifndef RENDERINGENGINE_RASTRTOOL_H
#define RENDERINGENGINE_RASTRTOOL_H

#include "LerpTool.h"

// 用于SH算法的两点线性插值
V2F lerpSH(const V2F &v1, const V2F &v2, const float t) {
    V2F r;
    r.clipPosi = v1.clipPosi * (1 - t) + v2.clipPosi * t;
    r.worldPosi = v1.worldPosi * (1 - t) + v2.worldPosi * t;
    r.normal = v1.normal * (1 - t) + v2.normal * t;
    r.uv = v1.uv * (1 - t) + v2.uv * t;
    r.invW = 1 / r.clipPosi[3];  // 1/w不能线性插值
    return r;
}

// 数值线形填充
float lerp(const float &n1, const float &n2, const float &t) {
    return n1 * (1 - t) + n2 * t;
}

// Pixel线性插值,用于片元着色
Pixel lerp(const Pixel& p1, const Pixel& p2, const float t) {
    // 将 uint8_t 转换为 float，以便计算
    const float r = static_cast<float>(p1.r) + (static_cast<float>(p2.r) - static_cast<float>(p1.r)) * t;
    const float g = static_cast<float>(p1.g) + (static_cast<float>(p2.g) - static_cast<float>(p1.g)) * t;
    const float b = static_cast<float>(p1.b) + (static_cast<float>(p2.b) - static_cast<float>(p1.b)) * t;
    const float a = static_cast<float>(p1.a) + (static_cast<float>(p2.a) - static_cast<float>(p1.a)) * t;

    // 确保结果落在 [0, 255] 范围内，并转换回 uint8_t
    return {static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(a, 0.0f, 255.0f))};
}

FloatPixel lerp(const FloatPixel& p1, const FloatPixel& p2, const float t) {
    const float r = p1.r + (p2.r - p1.r)*t;
    const float g = p1.g + (p2.g - p1.g)*t;
    const float b = p1.b + (p2.b - p1.b)*t;
    return {r, g, b};
}

V2F lerpNoLinear(const V2F &v1, const V2F &v2, const float t) {
    V2F r;
    // 透视矫正
    const float invW = lerp(v1.invW, v2.invW, t);

    // v1系数为(1-t), v2系数为t
    r.worldPosi = lerp(v1.worldPosi*v1.invW, v2.worldPosi*v2.invW, t) / invW;
    r.clipPosi = lerp(v1.clipPosi, v2.clipPosi, t);  // Z分量无需矫正，单用于深度测试的时候需要还原
    // NDC空间的深度值
    // r.NDCdepthNormal = lerp(v1.NDCdepthNormal*v1.invW, v2.NDCdepthNormal*v2.invW, t) / invW;
    r.normal = lerp(v1.normal*v1.invW, v2.normal*v2.invW, t) / invW;
    // r.normal = normalize(lerp(v1.normal*v1.invW, v2.normal*v2.invW, t) / invW);
    r.uv = lerp(v1.uv*v1.invW, v2.uv*v2.invW, t) / invW;
    r.invW = invW;
    return r;
}
#endif //RENDERINGENGINE_RASTRTOOL_H