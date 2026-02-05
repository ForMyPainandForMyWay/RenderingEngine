//
// Created by yyd on 2026/2/3.
//

#ifndef RENDERINGENGINE_DATAPACKEGGPU_CUH
#define RENDERINGENGINE_DATAPACKEGGPU_CUH

#include "BVH.cuh"
#include "Shape.cuh"

struct BVHNodeGPU;
struct BLASGPU;
struct MaterialGPU;
struct TextureMap;
struct SubMeshGPU;
struct MeshGPU;

struct CameraDataGPU {
    float Asp;
    float Scale;
    float4 cameraPos;
    Mat4GPU cameraRot;
    size_t width;
    size_t height;
    size_t pixelCount;
};

struct BVHDataGPU {
    BLASGPU* blas;
    uint32_t* BlasTriGPU;
    size_t triNums;
    BVHNodeGPU* BlasNodesGPU;
    size_t nodeNums;
    size_t blasCount;
    TLASGPU tlas;
};

// VBO EBO MeshesGPU SubMeshesGPU MaterialsGPU texPixelsGPU
struct ScenceDataGPU {
    MaterialGPU* MatGPU;
    size_t MaterialCount;
    VertexGPU* vboGPU;
    size_t vboCount;
    uint32_t* eboGPU;
    size_t eboCount;
    MeshGPU* MeshesGPU;
    size_t meshCount;
    SubMeshGPU* SubMeshesGPU;
    size_t subMeshCount;
    std::unordered_map<std::shared_ptr<TextureMap>, int> TextMap;
    size_t texPixelCount;
};

#endif //RENDERINGENGINE_DATAPACKEGGPU_CUH