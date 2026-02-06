//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_SHAPE_H
#define UNTITLED_SHAPE_H

#include <vector>

#include "V2F.hpp"
#include "Vec.hpp"


struct Pixel;
struct Vertex;
struct Triangle;


// 空间点
struct Vertex {
    Vec3 position;  // 物理坐标
    Vec3 normal;    // 法向量
    VecN<2> uv;        // 纹理坐标
    Pixel pix{};       // 顶点颜色

    bool operator==(const Vertex &other) const {
        return position == other.position &&
                 normal == other.normal &&
                     uv == other.uv;
    }
    [[nodiscard]] Vec4 getHomoPosi() const;
    [[nodiscard]] Vec4 getHomoNormal() const;
};


// 空间三角形，组装图元时使用
struct Triangle {
    std::array<V2F, 3> vex;
    bool alive=true;  // 用于标记背面剔除、退化剔除
    V2F& operator[](const size_t i) {return vex[i];}
    const V2F& operator[](const size_t i) const {return vex[i];}
};

// 片元
struct Fragment {
    size_t x{}, y{};
    float depth{};
    Vec4 worldPosi;
    Vec3 normal;
    VecN<2> uv;
    Vec3 VexLightF;      // 顶点颜色(float, 用于PixLight)
    std::array<Vec3,3> PixLightOri;  // 逐像素光源朝向
    Vec3 MainLightOri;  // 主光源朝向
    Vec3 CameraOri;  // 相机朝向
    bool alive{};  // 用于标记深度测试
    // 可能需要存储Material指针，暂时不设置
    void drop() { alive = false; }
    void keep() { alive = true; }
};

// Obj文件处理时候的临时结构,存储顶点在subMesh的索引
struct ObjFace {
    std::vector<uint32_t> vertexIndices;
    uint32_t operator[](size_t i) const;
    void addVexIndex(uint32_t index);
};
#endif //UNTITLED_SHAPE_H