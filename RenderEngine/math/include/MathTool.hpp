//
// Created by 冬榆 on 2025/12/29.
//

#ifndef UNTITLED_MATHTOOL_HPP
#define UNTITLED_MATHTOOL_HPP

#include <vector>

#include "Vec.hpp"
#include "Mat.hpp"
#include "Shape.hpp"


struct TextureMap;
struct V2F;
struct Triangle;
struct Fragment;


// euler = [roll, pitch, yaw] （绕 X、Y、Z 轴旋转，单位：弧度）
Vec4 Euler2Quaternion(const Vec3 &euler);
// 凸多边形裁剪
std::vector<Triangle> splitPoly2Tri(const std::vector<V2F>& poly);
// 透视除法
void PersDiv(Triangle &tri);
// 三角形面积 不保证符号
float TriScreenArea2(const Triangle &tri);

// 对角阵求逆
template<size_t M>
MatMN<M, M> diagMatInverse(const MatMN<M, M> &mat) {
    MatMN<M, M> result{};
    for (size_t i = 0; i < M; i++) {
        result[i][i] = 1 / mat[i][i];
    }
    return result;
}

// 深度映射，将z从[-w,w]映射到[0,w]
void DepthMap(Triangle &tri);
#endif //UNTITLED_MATHTOOL_HPP
