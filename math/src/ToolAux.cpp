//
// Created by 冬榆 on 2025/12/31.
//

#include "MathTool.hpp"
#include "V2F.h"


// 用于判断裁剪空间(非DNC空间)的点是否完全在视锥体外部
bool IsOutSideClip(const V2F& p, const uint8_t plane) {
    const float w = 1.0f * p.position[3];
    const float x = p.position[0];
    const float y = p.position[1];
    const float z = p.position[2];

    switch (plane) {
        case 0: return x < -w;
        case 1: return x > w;
        case 2: return y < -w;
        case 3: return y > w;
        case 4: return z < -w;
        case 5: return z > w;
        default: return false;
    }
}

// 用于SH算法判断是否在Clip空间的平面内
bool Inside(const float* line, const VecN<4> &posi) {
    return line[0] * posi[0] + line[1] * posi[1] + line[2] * posi[2] + line[3] * posi[3] > -(1e-6);
}

// 用于SH裁剪算法，计算截断点
V2F Intersect(const V2F &last, const V2F &current,const float* line) {
    const float da = last.position[0] * line[0] + last.position[1] * line[1] +
               last.position[2] * line[2] + last.position[3] * line[3];
    const float db = current.position[0] * line[0] + current.position[1] * line[1] +
               current.position[2] * line[2] + current.position[3] * line[3];
    const float weight = da / (da - db);
    return lerp(last, current, weight);
}

// 两点线性插值
V2F lerp(const V2F &v1, const V2F &v2, const float t) {
    V2F r;
    r.position = v1.position * (1 - t) + v2.position * t;
    r.normal = v1.normal * (1 - t) + v2.normal * t;
    r.uv = v1.uv * (1 - t) + v2.uv * t;
    r.invW = v1.invW * (1 - t) + v2.invW * t;
    return r;
}