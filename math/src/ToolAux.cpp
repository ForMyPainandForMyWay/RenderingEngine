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

// 数值线形填充
float lerp(const float &n1, const float &n2, const float &t) {
    return n1 * (1 - t) + n2 * t;
}

// 水平填充
void rasterizeSpan(const V2F &left, const V2F &right, int y,
                   std::vector<Fragment> &out) {
    const int xStart = static_cast<int>(ceil(left.position[0]));
    const int xEnd = static_cast<int>(floor(right.position[0]));
    if (xStart > xEnd) return;
    const float dx = right.position[0] - left.position[0];
    for (int x = xStart; x <= xEnd; ++x) {
        float t = dx==0 ? 0.0f : (static_cast<float>(x)-left.position[0])/dx;
        Fragment frag;
        frag.x = x;
        frag.y = y;
        // 透视修正插值；
        const float rw = lerp(left.invW, right.invW, t);
        frag.uv = lerp(left.uv * left.invW, right.uv * right.invW, t) / rw;
        frag.normal = normalize(lerp(left.normal*left.invW, right.normal*right.invW, t) / rw);
        frag.depth = lerp(left.position[2], right.position[2], t);
        out.emplace_back(frag);
    }
}

// 平顶三角形填充
void fillFlatTop(V2F v0, V2F v1, V2F v2, std::vector<Fragment> &result) {
    EdgeStepper leftEdge(v0, v2);
    EdgeStepper rightEdge(v1, v2);
    int yStart = static_cast<int>(leftEdge.yStart);
    int yEnd = static_cast<int>(leftEdge.yEnd);
    if (yStart > yEnd) return;
    for (int y = yStart; y <= yEnd; ++y) {
        rasterizeSpan(leftEdge.cur, rightEdge.cur, y, result);
        leftEdge.stepOnce();
        rightEdge.stepOnce();
    }
}

// 平底三角形填充
void fillFlatBottom(V2F v0, V2F v1, V2F v2, std::vector<Fragment> &result) {
    EdgeStepper leftEdge(v0, v1);
    EdgeStepper rightEdge(v0, v2);
    int yStart = static_cast<int>(leftEdge.yStart);
    int yEnd = static_cast<int>(leftEdge.yEnd);
    if (yStart > yEnd) return;;
    for (int y = yStart; y <= yEnd; ++y) {
        rasterizeSpan(leftEdge.cur, rightEdge.cur, y, result);
        leftEdge.stepOnce();
        rightEdge.stepOnce();
    }
}


EdgeStepper::EdgeStepper(const V2F &vTop, const V2F &vBottom) {
    yStart = ceil(vTop.position[1]);
    yEnd   = floor(vBottom.position[1]);
    const float dy = vBottom.position[1] - vTop.position[1];
    dxdy = (vBottom.position[0] - vTop.position[0]) / dy;
    cur = vTop;
    step.position = (vBottom.position - vTop.position) / dy;
    step.uv = (vBottom.uv - vTop.uv) / dy;
    step.normal = (vBottom.normal - vTop.normal) / dy;
    step.invW = (vBottom.invW - vTop.invW) / dy;
    x = vTop.position[0];
}

void EdgeStepper::stepOnce() {
    x += dxdy;
    cur.position += step.position;
    cur.uv += step.uv;
    cur.normal += step.normal;
    cur.invW += step.invW;
}