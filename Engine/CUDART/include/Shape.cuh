//
// Created by yyd on 2026/2/2.
//

#ifndef RENDERINGENGINE_SHAPE_CUH
#define RENDERINGENGINE_SHAPE_CUH
#include <vector_types.h>

struct Vertex;

struct VertexGPU {
    float3 position;
    float3 normal;
    float2 texCoord;
};

// 为了方便转换，这里也使用同样的内存布局
struct F2PGPU {
    size_t x{}, y{};
    float4 Albedo = {1.f, 1.f, 1.f, 1.f};
    float depth = 0;
    bool alive = true;
};


// 顶点转换
void Vertex2GPU(const Vertex& vex, VertexGPU& vexGPU);

#endif //RENDERINGENGINE_SHAPE_CUH