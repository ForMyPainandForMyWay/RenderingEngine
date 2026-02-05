//
// Created by yyd on 2026/2/2.
//

#ifndef RENDERINGENGINE_MESH_CUH
#define RENDERINGENGINE_MESH_CUH
#include <cstdint>
#include <vector_types.h>

struct VertexGPU;

struct MaterialGPU {
    float3 Ke;  // 自发光
    size_t KdPixOffset;  // 纹理贴图索引(像素数据索引)
    size_t KdPixCount;  // 纹理贴图大小
    int KdMapId;  // 纹理贴图索引,代替指针,-1表示空。纹理存储为cudaArray，所有纹理线性排列
};

struct SubMeshGPU {
    uint32_t SubEBOffset;  // 相对于Mesh的EBO索引（绝对索引）
    uint32_t SubEBOCount;
    int MaterialId;   // 代替材质指针的索引
};

struct MeshGPU {
    uint32_t MeshVBOffset;  // Vertex索引
    uint32_t MeshVBOCount;
    uint32_t MeshEBOffset;  // 边索引
    uint32_t MeshEBOCount;
    uint32_t MeshSubOffset;  // SubMesh索引
    uint32_t MeshSubCount;
};


#endif //RENDERINGENGINE_MESH_CUH