//
// Created by 冬榆 on 2026/1/7.
//

#include "RasterTool.h"
#include "LerpTool.h"
#include "Shape.h"


// 屏幕坐标三角形退化检测：三点面积是否为0，注意第一个的误差项是0.5,第二个误差项是1.0
void DegenerateClip(Triangle &tri) {
    float y0 = tri[0].position[1];
    float y1 = tri[1].position[1];
    float y2 = tri[2].position[1];

    // 简单退化
    if (std::max({y0, y1, y2}) - std::min({y0, y1, y2}) < 0.5f){
        tri.alive = false;
        return;
    }
    // 叉乘求面积,cross2D函数自动利用低二维计算叉乘
    if (fabs( crossInLow2D(tri[1].position-tri[0].position, tri[2].position-tri[0].position)) < 1.0f) {
        tri.alive = false;
    }
}

// 用于扫描线的三角形排序，不考虑退化情况（注意对应原点左上的屏幕坐标系）
void sortTriangle(Triangle &tri) {
    auto& v0 = tri[0];
    auto& v1 = tri[1];
    auto& v2 = tri[2];
    auto cmp = [](const V2F& a, const V2F& b) {
        if (fabs(a.position[1] - b.position[1]) > 1e-6) {
            return a.position[1] < b.position[1];}  // y 小在上
        return a.position[0] < b.position[0];       // y 相同，x 小在左
    };
    if (!cmp(v0, v1)) std::swap(v0, v1);
    if (!cmp(v1, v2)) std::swap(v1, v2);
    if (!cmp(v0, v1)) std::swap(v0, v1);
}

// 扫描线算法，传入拍好序的三角形
void ScanLine(const Triangle &sortedTri, std::vector<Fragment> &result) {
    V2F v0 = sortedTri[0];
    V2F v1 = sortedTri[1];
    V2F v2 = sortedTri[2];
    if (fabs(v1.position[1] - v0.position[1]) < 1e-4) {
        fillFlatTop(v0, v1, v2, result);
    } else if (fabs(v1.position[1] - v2.position[1]) < 1e-4) {
        fillFlatBottom(v0, v1, v2, result);
    } else {
        const float t = (v1.position[1] - v0.position[1]) / (v2.position[1] - v0.position[1]);
        const V2F vi = lerp(v0, v2, t);
        fillFlatBottom(v0, v1, vi, result);  // 上半部分，平底
        fillFlatTop(v1, vi, v2, result);     // 下半部分，平顶
    }
}

// 水平填充
void rasterizeSpan(const V2F &left, const V2F &right, const int y,
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
        frag.depth = lerp(left.position[2] * left.invW, right.position[2] * right.invW, t) / rw;
        out.emplace_back(frag);
    }
}

// 平顶三角形填充
void fillFlatTop(V2F v0, V2F v1, V2F v2, std::vector<Fragment> &result) {
    if (v0.position[0] > v1.position[0])
        std::swap(v0, v1);
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
    if (v1.position[0] > v2.position[0])
        std::swap(v1, v2);
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
    // cur.position += step.position;
    cur.position[0] = x;
    cur.position[1] += 1.0f;
    cur.position[2] += step.position[2];
    cur.position[3] += step.position[3];
    cur.uv += step.uv;
    cur.normal += step.normal;
    cur.invW += step.invW;
}