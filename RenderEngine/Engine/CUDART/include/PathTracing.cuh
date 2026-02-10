//
// Created by yyd on 2026/2/3.
//

#ifndef RENDERINGENGINE_PATHTRACING_CUH
#define RENDERINGENGINE_PATHTRACING_CUH
#include "DATAPackegGPU.cuh"

// 这里的宏定义用来控制extern的可见性
#ifdef CONSTANT_DEFINE
    #define CONST
#else
    #define CONST extern
#endif

CONST __constant__ CameraDataGPU CmDataGPu;
CONST __constant__ InstanceGPU TlasInstanceGPU[128];  // TLAS实例数量需要控制上限
CONST __constant__ BVHNodeGPU TlasNodesGPU[256];
CONST __constant__ cudaTextureObject_t texObjsGPU[128];  // 纹理描述符
CONST __constant__ size_t tlasNodeNums;
CONST __constant__ size_t tlasInstanceNums;

CONST __device__ BLASGPU* blasGPU;
CONST __device__ uint32_t* BlasTriGPU;  // 三角形的相对索引，使用时需要加上blas的offset
CONST __device__ BVHNodeGPU* BlasNodesGPU;
// Mesh相关的数据
CONST __device__ VertexGPU* vboGPU;
CONST __device__ uint32_t* eboGPU;
CONST __device__ MeshGPU* meshesGPU;
CONST __device__ SubMeshGPU* subMeshesGPU;
CONST __device__ MaterialGPU* materialsGPU;


__global__ void pathTracing(F2PGPU* resultGPU, int SPP=1, int maxDepth=8);
#endif //RENDERINGENGINE_PATHTRACING_CUH