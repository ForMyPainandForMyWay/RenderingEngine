//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_SHAPE_H
#define UNTITLED_SHAPE_H
#include <vector>
#include "Vec.hpp"

struct Vertex;
struct Triangle;



// 空间点
struct Vertex {
    VecN<3> position;  // 物理坐标
    VecN<3> normal;    // 法向量
    VecN<2> uv;        // 纹理坐标

    bool operator==(const Vertex &other) const {
        return position == other.position &&
                normal == other.normal &&
                uv == other.uv;
    }
};


// 空间三角形，组装图元时使用
struct Triangle {
    size_t vex[3]{};  // 顶点索引

    // 渲染阶段常用缓存
    VecN<3> screenPos[3];   // 屏幕坐标
    float depth[3]{};      // 深度
};


// Obj文件处理时候的临时结构,存储顶点在subMesh的索引
struct ObjFace {
    std::vector<uint32_t> vertexIndices;
    uint32_t operator[](size_t i) const;
    void addVexIndex(uint32_t index);
};
#endif //UNTITLED_SHAPE_H