//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_MESH_H
#define UNTITLED_MESH_H

#include "Shape.h"

class Mesh {
public:
    std::vector<Vertex> vertices;     // 顶点表(暂时没用)
    std::vector<Triangle> triangles;  // 三角面
    Material* material;               // 材质，默认为nullptr
};


#endif //UNTITLED_MESH_H