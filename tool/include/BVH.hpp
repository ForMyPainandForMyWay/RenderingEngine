//
// Created by 冬榆 on 2026/1/28.
//
#ifndef RENDERINGENGINE_BVH_HPP
#define RENDERINGENGINE_BVH_HPP
#include <vector>

#include "AABB.hpp"
#include "Mat.hpp"

struct HitInfo;
struct Material;
struct Ray;
class Mesh;
struct AABB;

// 基础 BVH 节点（通用）
struct BVHNode {
    AABB bbox;         // 包围盒
    int leftChild{};   // 左子节点索引
    int rightChild{};  // 右子节点索引
    int contentRef{};  //如果是叶子，指向三角形索引(BLAS)或实例索引(TLAS)
    bool isLeaf{};
};

// BLAS: 针对具体 Mesh。Mesh设置一个属性，指向 BLAS
struct BLAS {
    Mesh* mesh;  // 用于反向查找Mesh
    std::vector<uint32_t> triangles;  // 本地空间的三角形下标索引，索引内容是EBO的三角形顶点索引序列中每个三角形的首个顶点下表
    std::vector<BVHNode> nodes;       // 本地空间的BVH树
    AABB rootBounds;                  // 本地空间的整体包围盒

    [[nodiscard]] std::optional<HitInfo> Intersect(const Ray& localRay, const float& tMaxLimit) const;
};

// 渲染实例: 针对RenderObj
struct Instance {
    int blasIdx;        // 指向哪个 BLAS
    Mat4 transform;     // Model 矩阵
    Mat4 invTransform;  // invModel 矩阵 (用于变换光线)
    AABB worldBounds;   // 变换后的包围盒 (用于构建 TLAS)
};

// TLAS: 针对整个场景,由引擎持有
struct TLAS {
    std::vector<Instance> instances;
    std::vector<BVHNode> nodes; // 世界空间的BVH树，叶子指向 instances 下标
};
#endif //RENDERINGENGINE_BVH_HPP