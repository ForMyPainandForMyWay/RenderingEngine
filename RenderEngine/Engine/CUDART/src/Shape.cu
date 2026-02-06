//
// Created by yyd on 2026/2/2.
//

#include "Shape.cuh"
#include "Shape.hpp"
#include "TransfomTool.cuh"

void Vertex2GPU(const Vertex& vex, VertexGPU& vexGPU){
    vexGPU.position = Vec3ToFloat3(vex.position);
    vexGPU.normal = Vec3ToFloat3(vex.normal);
    vexGPU.texCoord = Vec2ToFloat2(vex.uv);
}
