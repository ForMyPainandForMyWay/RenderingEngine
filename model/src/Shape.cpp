//
// Created by yyd on 2025/12/24.
//

#include "Shape.h"


// 返回齐次坐标
VecN<4> Vertex::getHomoIndex() const{
    return VecN<4>{position[0], position[1], position[2], 1};
}

// 返回齐次向量
VecN<4> Vertex::getHomoNormal() const {
    return VecN<4>{normal[0], normal[1], normal[2], 0};
}

Triangle::Triangle(const V2F &v1, const V2F &v2, const V2F &v3) {
    this->vex[0] = v1;
    this->vex[1] = v2;
    this->vex[2] = v3;
}

uint32_t ObjFace::operator[](const size_t i) const {
    return this->vertexIndices[i];
}

void ObjFace::addVexIndex(uint32_t index) {
    this->vertexIndices.emplace_back(index);
}
