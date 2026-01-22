//
// Created by 冬榆 on 2026/1/7.
//

#ifndef RENDERINGENGINE_CLIPTOOL_H
#define RENDERINGENGINE_CLIPTOOL_H

#include "Shape.h"
#include "V2F.h"

struct Triangle;

constexpr float ViewPlanes[6][4] = {
    {0,0,1,1},   // near
    {0,0,-1,1},  // far
    {1,0,0,1},   // left
    {0,1,0,1},   // top
    {-1,0,0,1},  // right
    {0,-1,0,1}   // bottom
};

// Clip裁剪相关函数和变量
bool IsOutSideClip(const V2F& p, uint8_t plane);
bool AllVertexOutside(const V2F &p1, const V2F &p2, const V2F &p3);
bool AllVertexInside(const V2F &p1, const V2F &p2, const V2F &p3);
bool Inside(const float* plane, const Vec4 &posi);
V2F Intersect(const V2F &last, const V2F &current,const float* line);
std::vector<Triangle> PolyClip(const V2F &p1, const V2F &p2, const V2F &p3);
void FaceClip(Triangle &tri);

#endif //RENDERINGENGINE_CLIPTOOL_H