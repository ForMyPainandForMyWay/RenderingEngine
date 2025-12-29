//
// Created by yyd on 2025/12/24.
//

#include "Shape.h"


// VecN<3> Triangle::getNormal() const {
    // const VecN<3> e1 = this->vex[0].position - this->vex[1].position;
    // const VecN<3> e2 = this->vex[0].position - this->vex[2].position;
    // return normalize(cross(e1, e2));
// }

uint32_t ObjFace::operator[](const size_t i) const {
    return this->vertexIndices[i];
}

void ObjFace::addVexIndex(uint32_t index) {
    this->vertexIndices.emplace_back(index);
}
