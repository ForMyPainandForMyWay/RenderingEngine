//
// Created by yyd on 2025/12/24.
//

#include "Shape.h"
#include "Vec.hpp"


// 组装三角形
Triangle makeTriangle(const Vertex &v0, const Vertex &v1, const Vertex &v2) {
    Triangle t;
    t.vex[0] = v0;
    t.vex[1] = v1;
    t.vex[2] = v2;
    return t;
}

// 扇形分割,多边形分割为三角形
void processPolygon(const std::vector<Vertex> &inVertex,
                        std::vector<Triangle> &triangles){
    if (inVertex.size() < 3) return;
    for (size_t i = 1; i < inVertex.size()-1; ++i) {
        Triangle t = makeTriangle(inVertex[0], inVertex[i], inVertex[i+1]);
        triangles.emplace_back(t);
    }
}

VecN<3> Triangle::getNormal() const {
    const VecN<3> e1 = this->vex[0].position - this->vex[1].position;
    const VecN<3> e2 = this->vex[0].position - this->vex[2].position;
    return normalize(cross(e1, e2));
}
