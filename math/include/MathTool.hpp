//
// Created by 冬榆 on 2025/12/29.
//

#ifndef UNTITLED_MATHTOOL_HPP
#define UNTITLED_MATHTOOL_HPP

#include "Vec.hpp"
#include "Mat.hpp"
#include <vector>

struct V2F;
struct Triangle;


// euler = [roll, pitch, yaw] （绕 X、Y、Z 轴旋转，单位：弧度）
VecN<4> Euler2Quaternion(const VecN<3> &euler);
// 线性插值函数
V2F lerp(const V2F &v1, const V2F &v2, float t);
// 凸多边形裁剪
std::vector<Triangle> splitPoly2Tri(const std::vector<V2F>& poly);
// 透视除法
void PersDiv(Triangle &tri);

// Clip裁剪相关函数
bool IsOutSideClip(const V2F& p, uint8_t plane);
bool AllVertexOutside(const V2F &p1, const V2F &p2, const V2F &p3);
bool AllVertexInside(const V2F &p1, const V2F &p2, const V2F &p3);
bool Inside(const float* line, const VecN<4> &posi);
V2F Intersect(const V2F &last, const V2F &current,const float* line);
std::vector<Triangle> PolyClip(const V2F &p1, const V2F &p2, const V2F &p3);
void FaceClip(Triangle &tri);
constexpr float ViewPlanes[6][4] = {
    {0,0,1,1},   // near
    {0,0,-1,1},  // far
    {1,0,0,1},   // left
    {0,1,0,1},   // top
    {-1,0,0,1},  // right
    {0,-1,0,1}   // bottom
};



// 对角阵求逆
template<size_t M>
MatMN<M, M> diagMatInverse(const MatMN<M, M> &mat) {
    MatMN<M, M> result{};
    for (size_t i = 0; i < M; i++) {
        result[i][i] = 1 / mat[i][i];
    }
    return result;
}
#endif //UNTITLED_MATHTOOL_HPP
