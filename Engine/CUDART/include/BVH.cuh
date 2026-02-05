//
// Created by yyd on 2026/2/2.
//

#ifndef RENDERINGENGINE_BVH_CUH
#define RENDERINGENGINE_BVH_CUH
#include <cstdint>
#include <thrust/device_vector.h>

#include "AABB.cuh"
#include "Math.cuh"

struct VertexGPU;
struct SubMeshGPU;
struct MeshGPU;
struct Ray;
struct HitInfo;
struct Mat4GPU;

// 为了可以内存对齐，需要与CPU版本结构一致，同时注意AABBGPU的结构也要一致
struct BVHNodeGPU {
    AABBGPU bbox;         // 包围盒
    int leftChild;   // 左子节点索引
    int rightChild;  // 右子节点索引
    int contentRef;  //如果是叶子，指向三角形索引(BLAS)或实例索引(TLAS)
    bool isLeaf;
};

struct BLASGPU {
    uint32_t MeshGPUId;  // 用于反向查找Mesh GPU
    AABBGPU rootBounds;
    // 下面的是绝对索引，对应区域存储的是相对索引，调用数据时需要加上offset
    uint32_t triangleOffset;
    uint32_t triangleCount;
    uint32_t nodeOffset;
    uint32_t nodeCount;

    __device__  HitInfo Intersect(const Ray& localRay, const float& tMaxLimit) const;
};

// 注意内存结构需要和CPU版本一致
struct InstanceGPU {
    int blasIdx;
    Mat4GPU transform;
    Mat4GPU invTransform;
    AABBGPU worldBounds;
};

struct TLASGPU {
    InstanceGPU* instances;
    uint32_t instanceCount;
    BVHNodeGPU* nodes;
    uint32_t nodeCount;
};

#endif //RENDERINGENGINE_BVH_CUH