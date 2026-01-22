//
// Created by 冬榆 on 2025/12/29.
//

#include "MathTool.hpp"
#include "Shape.h"
#include "V2F.h"


Vec4 Euler2Quaternion(const Vec3 &euler) {
    constexpr auto tf = 3.14159265f / 180.0f;

    const float roll  = euler[0] * tf;
    const float pitch = euler[1] * tf;
    const float yaw   = euler[2] * tf;

    // 预计算半角
    const float cy = cosf(yaw * 0.5f);
    const float sy = sinf(yaw * 0.5f);
    const float cp = cosf(pitch * 0.5f);
    const float sp = sinf(pitch * 0.5f);
    const float cr = cosf(roll * 0.5f);
    const float sr = sinf(roll * 0.5f);

    Vec4 q;
    q[3] = cr * cp * cy + sr * sp * sy;  // w
    q[0] = sr * cp * cy - cr * sp * sy;  // x
    q[1] = cr * sp * cy + sr * cp * sy;  // y
    q[2] = cr * cp * sy - sr * sp * cy;  // z

    return q;
}

std::vector<Triangle> splitPoly2Tri(const std::vector<V2F>& poly) {
    std::vector<Triangle> result;
    for (auto i = 2; i < poly.size(); ++i)
        result.emplace_back(Triangle{poly[0], poly[i-1], poly[i]});
    return result;
}

void PersDiv(Triangle &tri) {
    for (uint8_t i = 0; i < 3; i++) {
        tri[i].clipPosi *= tri[i].invW;
        // 由于shader阶段不再提前乘w，这里不再除一个w
        // tri[i].normal *= tri[i].invW;
        // tri[i].uv *= tri[i].invW;
    }
}

float TriScreenArea2(const Triangle &tri) {
    const float e1x = tri[1].clipPosi[0] - tri[0].clipPosi[0];
    const float e1y= tri[1].clipPosi[1] - tri[0].clipPosi[1];
    const float e2x = tri[2].clipPosi[0] - tri[0].clipPosi[0];
    const float e2y = tri[2].clipPosi[1] - tri[0].clipPosi[1];
    return e1x * e2y - e1y * e2x;
}

// 深度映射，将z从[-w,w]映射到[0,w]
void DepthMap(Triangle &tri) {
    if (!tri.alive) return;
    for (size_t i = 0; i < 3; ++i) {
        tri[i].clipPosi[2] = (tri[i].clipPosi[2] + tri[i].clipPosi[3]) * 0.5f;
    }
}