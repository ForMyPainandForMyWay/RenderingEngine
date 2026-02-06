//
// Created by 冬榆 on 2026/1/7.
//

#include "ClipTool.hpp"
#include "LerpTool.hpp"
#include "MathTool.hpp"


// 用于判断裁剪空间(非DNC空间)的点是否完全在视锥体外部
bool IsOutSideClip(const V2F& p, const uint8_t plane) {
    const float w = 1.0f * p.clipPosi[3];
    const float x = p.clipPosi[0];
    const float y = p.clipPosi[1];
    const float z = p.clipPosi[2];

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

// 用于SH算法判断是否在Clip空间的平面内
bool Inside(const float* plane, const Vec4 &posi) {
    // Ax + By + Cz + Dw >= 0 即为在平面内部
    const float val = plane[0]*posi[0] + plane[1]*posi[1] + plane[2]*posi[2] + plane[3]*posi[3];
    return val >= -1e-4f; // 加小 epsilon 防止浮点误差
}

// 用于SH裁剪算法，计算截断点
V2F Intersect(const V2F &last, const V2F &current, const float* line) {
    const float da = last.clipPosi[0] * line[0] + last.clipPosi[1] * line[1] +
               last.clipPosi[2] * line[2] + last.clipPosi[3] * line[3];
    const float db = current.clipPosi[0] * line[0] + current.clipPosi[1] * line[1] +
               current.clipPosi[2] * line[2] + current.clipPosi[3] * line[3];
    const float weight = da / (da - db);
    return lerpSH(last, current, weight);
}

// SH算法，需要保证传入所有的点都在裁剪体内.返回切分后的三角形序列
std::vector<Triangle> PolyClip(const V2F &p1, const V2F &p2, const V2F &p3) {
    std::vector output = { p1, p2, p3 };
    for (const auto ViewPlane : ViewPlanes) {
        std::vector<V2F> input = output;
        output.clear();
        for (size_t j = 0; j < input.size(); ++j) {
            const V2F &current = input[j];
            const V2F &last = input[(j + input.size() - 1) % input.size()];
            const bool currInside = Inside(ViewPlane, current.clipPosi);
            const bool lastInside = Inside(ViewPlane, last.clipPosi);
            if (currInside) {
                if (!lastInside) {
                    output.push_back(Intersect(last, current, ViewPlane));}
                output.push_back(current);
            } else if (lastInside) {
                output.push_back(Intersect(last, current, ViewPlane));}
        }
        // 完全被裁掉
        if (output.empty()) break;
    }
    return splitPoly2Tri(output);
}

// NDC空间面剔除，逆时针的三角是正向三角(true)，顺时针三角需要剔除(false)
void FaceClip(Triangle &tri) {
    tri.alive = TriScreenArea2(tri) > -1e-3f;
}