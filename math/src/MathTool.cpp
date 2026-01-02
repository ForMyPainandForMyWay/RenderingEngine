//
// Created by 冬榆 on 2025/12/29.
//

#include "MathTool.hpp"
#include "Shape.h"
#include "V2F.h"


VecN<4> Euler2Quaternion(const VecN<3> &euler) {
    const float roll  = euler[0];
    const float pitch = euler[1];
    const float yaw   = euler[2];

    // 预计算半角
    const float cy = cosf(yaw * 0.5f);
    const float sy = sinf(yaw * 0.5f);
    const float cp = cosf(pitch * 0.5f);
    const float sp = sinf(pitch * 0.5f);
    const float cr = cosf(roll * 0.5f);
    const float sr = sinf(roll * 0.5f);

    VecN<4> q;
    q[0] = cr * cp * cy + sr * sp * sy;  // w
    q[1] = sr * cp * cy - cr * sp * sy;  // x
    q[2] = cr * sp * cy + sr * cp * sy;  // y
    q[3] = cr * cp * sy - sr * sp * cy;  // z

    return q;
}

// 验证所有Clip顶点都在视锥体外部
bool AllVertexOutside(const V2F &p1, const V2F &p2, const V2F &p3) {
    for (uint8_t plane = 0; plane < 6; ++plane) {
        if (IsOutSideClip(p1, plane) &&
            IsOutSideClip(p2, plane) &&
            IsOutSideClip(p3, plane))
            return true;
    }
    return false;
}

// 用于验证所有Clip顶点都在视锥体内部
bool AllVertexInside(const V2F &p1, const V2F &p2, const V2F &p3) {
    for (uint8_t plane = 0; plane < 6; ++plane) {
        if (IsOutSideClip(p1, plane) ||
            IsOutSideClip(p2, plane) ||
            IsOutSideClip(p3, plane))
            return false;
    }
    return true;
}

// SH算法，需要保证传入所有的点都在裁剪体内.返回切分后的三角形序列
std::vector<Triangle> PolyClip(const V2F &p1, const V2F &p2, const V2F &p3) {
    std::vector output = {p1, p2, p3};
    for (const auto ViewPlane : ViewPlanes) {
        std::vector input(output);
        output.clear();
        for (auto j = 0; j < input.size(); j++) {
            V2F current = input[j];
            V2F last = input[(j + input.size() - 1) % input.size()];

            if ( Inside(ViewPlane, current.position) ) {
                if (!Inside(ViewPlane, last.position)) {
                    V2F intersecting = Intersect(last, current, ViewPlane);
                    output.emplace_back(intersecting);
                }
            }
            else if ( Inside(ViewPlane, last.position)){
                V2F intersecting = Intersect(last, current, ViewPlane);
                output.emplace_back(intersecting);
            }
        }
    }
    return splitPoly2Tri(output);
}

std::vector<Triangle> splitPoly2Tri(const std::vector<V2F>& poly) {
    std::vector<Triangle> result;
    for (auto i = 1; i < poly.size(); ++i) {
        result.emplace_back(poly[0], poly[i], poly[i+1]);
    }
    return result;
}

void PersDiv(Triangle &tri) {
    for (uint8_t i = 0; i < 3; i++) {
        tri[i].position *= tri[i].invW;
        tri[i].normal *= tri[i].invW;
        tri[i].uv *= tri[i].invW;
    }
}

// NDC空间面剔除，逆时针的三角是正向三角(true)，顺时针三角需要剔除(false)
void FaceClip(Triangle &tri) {
    const auto e0 = tri[1].position - tri[0].position;
    const auto e1 = tri[2].position - tri[0].position;
    tri.alive = crossInLow2D(e0, e1) > 0.0f;
}

// 屏幕坐标三角形退化检测：三点面积是否为0，注意第一个的误差项是0.5,第二个误差项是1.0
bool TriangleIsAlive(const Triangle &tri) {
    float y0 = tri[0].position[1];
    float y1 = tri[1].position[1];
    float y2 = tri[2].position[1];

    // 简单退化
    if (std::max({y0, y1, y2}) - std::min({y0, y1, y2}) < 0.5f)
        return false;
    // 叉乘求面积,cross2D函数自动利用低二维计算叉乘
    if (fabs( crossInLow2D(tri[1].position-tri[0].position, tri[2].position-tri[0].position)) < 1.0f) {
        return false;
    }
    return true;
}

// 三角形排序，不考虑退化情况
void sortTriangle(Triangle &tri) {
    V2F v0 = tri[0];
    V2F v1 = tri[1];
    V2F v2 = tri[2];
    if (v0.position[1] > v1.position[1]) std::swap(v0, v1);
    if (v1.position[1] > v2.position[1]) std::swap(v1, v2);
    if (v0.position[1] > v1.position[1]) std::swap(v0, v1);
    tri[0] = v0;
    tri[1] = v1;
    tri[2] = v2;
}

// 扫描线算法，传入拍好序的三角形
void ScanLine(const Triangle &sortedTri, std::vector<Fragment> &result) {
    V2F v0 = sortedTri[0];
    V2F v1 = sortedTri[1];
    V2F v2 = sortedTri[2];
    if (fabs(v1.position[1] - v0.position[1]) < 1e-6) {
        fillFlatTop(v0, v1, v2, result);
    } else if (fabs(v1.position[1] - v2.position[1]) < 1e-6) {
        fillFlatBottom(v0, v1, v2, result);
    } else {
        const float t = (v1.position[1] - v0.position[1]) / (v2.position[1] - v0.position[1]);
        const V2F vi = lerp(v0, v2, t);
        fillFlatTop(v0, v1, vi, result);
        fillFlatBottom(v1, vi, v2, result);
    }
}

