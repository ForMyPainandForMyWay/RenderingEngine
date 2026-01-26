//
// Created by yyd on 2025/12/24.
//

#include "Shape.h"


// 返回齐次坐标
Vec4 Vertex::getHomoPosi() const{
    return Vec4{position[0], position[1], position[2], 1};
}

// 返回齐次向量
Vec4 Vertex::getHomoNormal() const {
    return Vec4{normal[0], normal[1], normal[2], 0};
}

uint32_t ObjFace::operator[](const size_t i) const {
    return this->vertexIndices[i];
}

void ObjFace::addVexIndex(uint32_t index) {
    this->vertexIndices.emplace_back(index);
}
