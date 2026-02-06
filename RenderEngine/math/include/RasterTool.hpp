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

// 重心坐标
void BarycentricOptimizedFull(Triangle& tri, std::vector<Fragment>& result, int screenWidth, int screenHeight);
void Scanline(Triangle& tri, std::vector<Fragment>& result, int screenWidth, int screenHeight);
#endif //RENDERINGENGINE_RASTERTOOL_H