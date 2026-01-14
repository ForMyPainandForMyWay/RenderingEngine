//
// Created by 冬榆 on 2026/1/7.
//

#ifndef RENDERINGENGINE_RASTERTOOL_H
#define RENDERINGENGINE_RASTERTOOL_H

#include <vector>

struct Triangle;
struct Fragment;

// 光栅化工具
void DegenerateClip(Triangle &tri);
void sortTriangle(Triangle &tri);

// 重心坐标
void Barycentric(Triangle &tri, std::vector<Fragment> &result);

#endif //RENDERINGENGINE_RASTERTOOL_H