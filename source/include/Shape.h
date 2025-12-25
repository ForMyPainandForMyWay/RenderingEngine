//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_SHAPE_H
#define UNTITLED_SHAPE_H
#include <string>
#include <vector>
#include "Vec.hpp"

struct Vertex;
struct Triangle;
struct Material;


// 表面材质
struct Material {
    std::string name;
    VecN<3> Ka;   // 环境光
    VecN<3> Kd;   // 漫反射
    VecN<3> Ks;   // 高光
    float Ns;  // 光泽指数
    std::string map_Kd; // 漫反射贴图
};


// 空间点
struct Vertex {
    VecN<3> position;  // 物理坐标
    VecN<3> normal;    // 法向量
    VecN<2> uv;   // 纹理坐标

    bool operator==(const Vertex &other) const {
        return position == other.position &&
                normal == other.normal &&
                uv == other.uv;
    }

};


// 空间三角形
struct Triangle {
    Vertex vex[3];  // 顶点

    // 渲染阶段常用缓存
    VecN<3> screenPos[3];   // 屏幕坐标
    float depth[3]{};      // 深度
    VecN<3> getNormal() const;    // 求平面法向量
};


Triangle makeTriangle(const Vertex &v0, const Vertex &v1, const Vertex &v2);
void processPolygon(const std::vector<Vertex> &inVerts,
                        std::vector<Triangle> &triangles);
#endif //UNTITLED_SHAPE_H