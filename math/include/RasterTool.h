//
// Created by 冬榆 on 2026/1/7.
//

#ifndef RENDERINGENGINE_RASTERTOOL_H
#define RENDERINGENGINE_RASTERTOOL_H

#include <vector>

#include "V2F.h"

struct Triangle;
struct Fragment;


// 光栅化工具
void DegenerateClip(Triangle &tri);
void sortTriangle(Triangle &tri);
void ScanLine(const Triangle &sortedTri, std::vector<Fragment> &result);
void rasterizeSpan(const V2F &left, const V2F &right, int y,
                   std::vector<Fragment> &out);
void fillFlatTop(V2F v0, V2F v1, V2F v2, std::vector<Fragment> &result);
void fillFlatBottom(V2F v0, V2F v1, V2F v2, std::vector<Fragment> &result);
struct EdgeStepper {
    float yStart{}, yEnd{};
    float x{}, dxdy{};
    V2F cur, step;  // 当前位置待插值点、每行的插值增量
    EdgeStepper(const V2F &vTop, const V2F &vBottom);
    void stepOnce();
};

#endif //RENDERINGENGINE_RASTERTOOL_H