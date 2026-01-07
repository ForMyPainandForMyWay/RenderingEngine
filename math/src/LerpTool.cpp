//
// Created by 冬榆 on 2025/12/31.
//

#include "LerpTool.h"
#include "V2F.h"

// 用于SH算法的两点线性插值
V2F lerpSH(const V2F &v1, const V2F &v2, const float t) {
    V2F r;
    r.position = v1.position * (1 - t) + v2.position * t;
    r.normal = v1.normal * (1 - t) + v2.normal * t;
    r.uv = v1.uv * (1 - t) + v2.uv * t;
    r.invW = 1 / r.position[3];  // 1/w不能线性插值
    return r;
}

// 两点线性插值(用于光栅化，这时候理论上已经用不到invW了)
V2F lerp(const V2F &v1, const V2F &v2, float t) {
    V2F r;
    r.position = v1.position * (1 - t) + v2.position * t;
    r.normal = v1.normal * (1 - t) + v2.normal * t;
    r.uv = v1.uv * (1 - t) + v2.uv * t;
    r.invW = v1.invW * (1 - t) + v2.invW * t;
    return r;
}

// 数值线形填充
float lerp(const float &n1, const float &n2, const float &t) {
    return n1 * (1 - t) + n2 * t;
}

// Pixel线性插值
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