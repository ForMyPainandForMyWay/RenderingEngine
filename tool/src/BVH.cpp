#include <vector>
#include <stack>
#include <algorithm>
#include <memory>

#include "AABB.hpp"
#include "Mesh.hpp"
#include "BVH.hpp"
#include "Engine.hpp"
#include "HitInfo.hpp"
#include "Ray.hpp"
#include "RayTraceTool.hpp"

// 辅助结构：预计算三角形信息
struct TriInfo {
    uint32_t eboIndex; // 原始 EBO 索引
    AABB bbox;         // 三角形自身的包围盒
    Vec3 centroid;     // 三角形质心
};

// 栈任务
struct BuildTask {
    int nodeIndex; // BLAS 节点数组下标
    int start;     // 三角形列表起始
    int end;       // 三角形列表结束
};

std::shared_ptr<BLAS> Mesh::BuildBLAS() {
    if (triIsEmpty()) return nullptr;
    auto blas = std::make_shared<BLAS>();
    blas->mesh = this;

    const size_t triCount = getTriNums();
    std::vector<TriInfo> triInfos;
    triInfos.reserve(triCount);
    for (size_t i = 0; i < triCount; ++i) {
        // 获取三角形顶点 (这里假设 Vertex 有 .pos 成员且能转为 Vec3)
        // 你可能需要根据实际 Vertex 结构强转，例如: *(Vec3*)&VBO[...]
        const Vec3& v1 = VBO[EBO[i * 3 + 0]].position;
        const Vec3& v2 = VBO[EBO[i * 3 + 1]].position;
        const Vec3& v3 = VBO[EBO[i * 3 + 2]].position;
        AABB triBox;
        triBox.grow(v1);
        triBox.grow(v2);
        triBox.grow(v3);
        const Vec3 centroid = (v1 + v2 + v3) * (1.0f / 3.0f);
        triInfos.push_back({ static_cast<uint32_t>(i * 3), triBox, centroid });
    }

    // 初始化构建栈
    blas->nodes.emplace_back(); // 创建 Root 节点
    std::stack<BuildTask> stack;
    stack.push({0, 0, static_cast<int>(triCount)});

    // 循环构建
    while (!stack.empty()) {
        const auto [nodeIndex, start, end] = stack.top();
        stack.pop();
        const int count = end - start;
        // 计算当前节点的整体 AABB
        AABB nodeBox;
        for (int i = start; i < end; ++i) {
            nodeBox.grow(triInfos[i].bbox);
        }
        blas->nodes[nodeIndex].bbox = nodeBox;
        // 叶子节点判定 (三角形数 <= 2)
        if (count <= 2) {
            blas->nodes[nodeIndex].isLeaf = true;
            blas->nodes[nodeIndex].contentRef = start; // 指向排序后的起始位置
            blas->nodes[nodeIndex].leftChild = -1;
            blas->nodes[nodeIndex].rightChild = count;
            continue;
        }

        // 寻找分割轴
        AABB centroidBounds;
        for (int i = start; i < end; ++i) {
            centroidBounds.grow(triInfos[i].centroid);
        }
        int axis = centroidBounds.maxDimension();
        float splitPos = centroidBounds.center()[axis];
        // 如果质心包围盒太小（所有三角形重心重合），强制平分以防死循环
        if (centroidBounds.extent()[axis] < 1e-6f) {
            splitPos = (centroidBounds.Pmin[axis] + centroidBounds.Pmax[axis]) * 0.5f;
        }
        const auto it = std::partition(triInfos.begin() + start, triInfos.begin() + end,
            [axis, splitPos](const TriInfo& tri) {
                return tri.centroid[axis] < splitPos;
            });
        int mid = static_cast<int>(std::distance(triInfos.begin(), it));

        // 一边为空时，强行中点切分
        if (mid == start || mid == end) {
            mid = start + count / 2;
        }
        // 创建子节点
        blas->nodes[nodeIndex].isLeaf = false;
        blas->nodes[nodeIndex].contentRef = -1;
        const int leftIdx = static_cast<int>(blas->nodes.size());
        blas->nodes.emplace_back();
        const int rightIdx = static_cast<int>(blas->nodes.size());
        blas->nodes.emplace_back();
        // 重新赋值，防止 vector 扩容导致引用失效
        blas->nodes[nodeIndex].leftChild = leftIdx;
        blas->nodes[nodeIndex].rightChild = rightIdx;
        // 右子树先入栈，左子树后入栈（保持 DFS 顺序）
        stack.push({rightIdx, mid, end});
        stack.push({leftIdx, start, mid});
    }

    // 写入最终数据
    blas->rootBounds = blas->nodes[0].bbox;
    blas->triangles.resize(triCount);
    for (size_t i = 0; i < triCount; ++i) {
        // 这里的 triInfos 直接按顺序写入 EBO 索引
        blas->triangles[i] = triInfos[i].eboIndex;
    }
    return blas;
}

// 对某个Mesh的BVH进行求交测试
std::optional<HitInfo> BLAS::Intersect(const Ray& localRay, const float& tMaxLimit) const{
    if (nodes.empty()) return std::nullopt;
    const Vec3 rayPos = {localRay.orignPosi[0], localRay.orignPosi[1], localRay.orignPosi[2]};
    Vec3 rayDir = {localRay.Direction[0], localRay.Direction[1], localRay.Direction[2]};
    const Vec3 rayDirInv = { 1.0f / rayDir[0], 1.0f / rayDir[1], 1.0f / rayDir[2] };
    float closestT = tMaxLimit;
    // 栈存储待遍历的节点索引
    int nodeStack[64];
    int stackPtr = 0;
    // 根节点压栈
    nodeStack[stackPtr++] = 0;
    std::optional<HitInfo> result = std::nullopt;
    while (stackPtr > 0) {
        // 弹出节点
        const int nodeIdx = nodeStack[--stackPtr];
        const BVHNode& node = nodes[nodeIdx];
        // AABB 测试 (剪枝)
        if (float tEntry; !IntersectAABB(node.bbox, rayPos, rayDirInv, closestT, tEntry)) {
            continue; // 不相交，或者比当前最近交点远
        }
        // 如果是叶子节点，进行精确三角形求交
        if (node.isLeaf) {
            const int triStart = node.contentRef;
            const int triCount = node.rightChild < 0 ? 1 : node.rightChild;
            for (int i = 0; i < triCount; ++i) {
                // 获取三角形在 Mesh EBO 中的起始索引
                const uint32_t eboIdx = this->triangles[triStart + i];
                const Vertex& v1 = mesh->getVertex(mesh->EBO[eboIdx + 0]);
                const Vertex& v2 = mesh->getVertex(mesh->EBO[eboIdx + 1]);
                const Vertex& v3 = mesh->getVertex(mesh->EBO[eboIdx + 2]);
                auto hit = MollerTrumbore(
                    v1, v2, v3,
                    rayPos,  // 射线起点 (Loca)
                    rayDir,  // 射线方向 (Local)
                    closestT
                );
                if (hit.has_value()) {
                    result = std::move(hit);
                    // 这里还需要补全material信息
                    for (const auto& subM : *mesh) {
                        const auto oft = subM.getOffset();
                        if (const auto end = subM.getIdxCount() + oft;
                            eboIdx >= oft && eboIdx < end) {
                            result->mat = subM.getMaterial();
                            break;
                        }
                    }
                }
            }
        }
        // 如果是内部节点，将子节点压栈
        else {
            const int leftChild = node.leftChild;
            const int rightChild = node.rightChild;
            float tLeft = 0, tRight = 0;
            const bool hitLeft = IntersectAABB(nodes[leftChild].bbox, rayPos, rayDirInv, closestT, tLeft);
            if (const bool hitRight = IntersectAABB(nodes[rightChild].bbox, rayPos, rayDirInv, closestT, tRight);
                hitLeft && hitRight) {
                if (tLeft < tRight) {
                    nodeStack[stackPtr++] = rightChild; // 远的先入栈（后处理）
                    nodeStack[stackPtr++] = leftChild;  // 近的后入栈（先处理）
                } else {
                    nodeStack[stackPtr++] = leftChild;
                    nodeStack[stackPtr++] = rightChild;
                }} else if (hitLeft) {
                    nodeStack[stackPtr++] = leftChild;
                } else if (hitRight) {
                    nodeStack[stackPtr++] = rightChild;
                }
            }
        }
    return result;
}

void Engine::BuildTLAS(const std::vector<uint16_t>& models) {
        tlas = std::make_unique<TLAS>();
        tlas->instances.reserve(models.size());
        for (auto& objIdx : models) {
            auto obj = renderObjs[objIdx];
            // 获取 Mesh 指针
            const auto mesh = obj.getMesh();
            if (!mesh || mesh->triIsEmpty()) continue; // 跳过空物体
            // 获取 BLAS 索引
            const size_t blasIdx = mesh->BLASIdx;
            if (blasIdx >= blasList.size()) continue;
            // 构造 Instance
            Instance inst;
            inst.blasIdx = static_cast<int>(blasIdx);
            inst.transform = obj.ModelMat();
            inst.invTransform = obj.invModelMat();
            // 计算 World AABB
            const AABB& localBounds = blasList[blasIdx]->rootBounds;
            inst.worldBounds = TransformAABB(localBounds, inst.transform);
            tlas->instances.push_back(inst);
        }
        struct BuildTask {
            int nodeIndex;
            int start; int end;
        };

        tlas->nodes.emplace_back(); // 根节点 index 0
        std::stack<BuildTask> stack;
        stack.push({0, 0, static_cast<int>(tlas->instances.size())});

        // 非递归构建循环
        while (!stack.empty()) {
            const auto [nodeIndex, start, end] = stack.top();
            stack.pop();
            // 获取当前节点引用 (注意 vector 扩容失效问题，这里用索引访问更安全)
            const int count = end - start;
            // 计算当前节点整体包围盒 (并集)
            AABB nodeBox;
            // 计算当前范围内所有 Instance 中心的包围盒用于划分
            AABB centBounds;
            for (int i = start; i < end; ++i) {
                const auto& inst = tlas->instances[i];
                nodeBox.grow(inst.worldBounds);
                centBounds.grow(inst.worldBounds.center());
            }
            tlas->nodes[nodeIndex].bbox = nodeBox;
            // 叶子节点判断
            // 对于 TLAS，当只剩 1 个实例时才停止
            if (count == 1) {
                tlas->nodes[nodeIndex].isLeaf = true;
                tlas->nodes[nodeIndex].contentRef = start; // 指向 instances 的下标
                tlas->nodes[nodeIndex].leftChild = -1;
                tlas->nodes[nodeIndex].rightChild = -1;
                continue;
            }
            // 寻找分割轴 (实例中心的最长轴)
            int axis = centBounds.maxDimension();
            float splitPos = centBounds.center()[axis];
            // 如果所有实例中心重合，无法分割
            if (centBounds.extent()[axis] < 1e-5f) {
                // 强制平分
                 splitPos = (centBounds.Pmin[axis] + centBounds.Pmax[axis]) * 0.5f;
            }
            // 执行划分
            // 对 tlas.instances 进行重排
            const auto it =
                std::partition(tlas->instances.begin() + start, tlas->instances.begin() + end,
                [axis, splitPos](const Instance& inst) {
                    return inst.worldBounds.center()[axis] < splitPos;
                });

            int mid = static_cast<int>(std::distance(tlas->instances.begin(), it));
            // 如果只有两个完全重合的物体，强制从中间切开
            if (mid == start || mid == end) {
                mid = start + count / 2;
            }
            // 创建子节点
            tlas->nodes[nodeIndex].isLeaf = false;
            tlas->nodes[nodeIndex].contentRef = -1;

            const int leftIdx = static_cast<int>(tlas->nodes.size());
            tlas->nodes.emplace_back();
            const int rightIdx = static_cast<int>(tlas->nodes.size());
            tlas->nodes.emplace_back();
            tlas->nodes[nodeIndex].leftChild = leftIdx;
            tlas->nodes[nodeIndex].rightChild = rightIdx;
            // 右孩子先入栈，左孩子后入栈
            stack.push({rightIdx, mid, end});
            stack.push({leftIdx, start, mid});
        }
}