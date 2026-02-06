//
// Created by yyd on 2026/2/2.
//

#include "BVH.cuh"
#include "Ray.cuh"
#include "HitTool.cuh"
#include "Mesh.cuh"
#include "PathTracing.cuh"

// 对blas进行求交测试
__device__  HitInfo BLASGPU::Intersect(
    const Ray& localRay, const float& tMaxLimit) const {
    HitInfo hitInfo;
    hitInfo.Valid = false;
    if (tlasNodeNums == 0) return hitInfo;
    const float3 rayPos = {localRay.orignPosi.x, localRay.orignPosi.y, localRay.orignPosi.z};
    float3 rayDir = {localRay.Direction.x, localRay.Direction.y, localRay.Direction.z};
    const float3 rayDirInv = { 1.0f / rayDir.x, 1.0f / rayDir.y, 1.0f / rayDir.z };
    float closestT = tMaxLimit;
    // 栈存储待遍历的节点索引
    int nodeStack[64];
    int stackPtr = 0;
    // 根节点压栈
    nodeStack[stackPtr++] = 0;
    HitInfo result{};
    result.Valid = false;
    while (stackPtr > 0) {
        // 弹出节点
        const int nodeIdx = nodeStack[--stackPtr] + nodeOffset;  // 注意这里要加偏移，BlasNodesGPU是全局的
        const auto& node = BlasNodesGPU[nodeIdx];
        // AABB 测试 (剪枝)
        if (float tEntry; !IntersectAABBGPU(node.bbox, rayPos, rayDirInv, closestT, tEntry)) {
            continue; // 不相交，或者比当前最近交点远
        }
        // 如果是叶子节点，进行精确三角形求交
        // 先得到Mesh索引和对应偏移
        const auto meshId = this->MeshGPUId;
        const MeshGPU& mesh = meshesGPU[meshId];
        const auto EBOft = mesh.MeshEBOffset;
        const auto VBOft = mesh.MeshVBOffset;
        if (node.isLeaf) {
            // 注意这里的 索引 要转换成 相对索引 + offset
            const int triStart = triangleOffset + node.contentRef;
            const int triCount = node.rightChild < 0 ? 1 : node.rightChild;
            for (int i = 0; i < triCount; ++i) {
                // 获取三角形在 Mesh EBO 中的起始索引
                // 注意这里triStart应该已经转换成 绝对索引，但是eboIdx是相对索引，还要进行转换 + EBOft
                const uint32_t eboIdx = EBOft + BlasTriGPU[triStart + i];
                const VertexGPU v1 = vboGPU[VBOft + eboGPU[eboIdx + 0]];
                const VertexGPU v2 = vboGPU[VBOft + eboGPU[eboIdx + 1]];
                const VertexGPU v3 = vboGPU[VBOft + eboGPU[eboIdx + 2]];
                auto hit = MollerTrumbore(
                    v1, v2, v3,
                    rayPos, rayDir,closestT
                );
                if (hit.Valid) {
                    result = hit;
                    // 这里还需要补全material信息
                    for (size_t SubI = mesh.MeshSubOffset; SubI < mesh.MeshSubOffset + mesh.MeshSubCount; SubI++) {
                        const auto& [SubEBOffset, SubEBOCount, MaterialId] =
                            subMeshesGPU[SubI];
                        const auto oft = SubEBOffset;
                        if (const auto end = oft + SubEBOCount;
                            eboIdx >= oft && eboIdx < end) {
                            result.matID = MaterialId;
                        }
                    }
                }
            }
        }
        // 如果是内部节点，将子节点压栈
        else {
            // 这里也要转换为 绝对索引
            const int leftChild = node.leftChild + nodeOffset;
            const int rightChild = node.rightChild + nodeOffset;
            float tLeft = 0, tRight = 0;
            const bool hitLeft = IntersectAABBGPU(
                BlasNodesGPU[leftChild].bbox,
                rayPos,
                rayDirInv,
                closestT,
                tLeft);
            if (const bool hitRight =
                    IntersectAABBGPU(BlasNodesGPU[rightChild].bbox,
                    rayPos,
                    rayDirInv,
                    closestT,
                    tRight);
                hitLeft && hitRight) {
                if (tLeft < tRight) {
                    nodeStack[stackPtr++] = rightChild - nodeOffset; // 远的先入栈
                    nodeStack[stackPtr++] = leftChild - nodeOffset;  // 近的后入栈
                } else {
                    nodeStack[stackPtr++] = leftChild - nodeOffset;
                    nodeStack[stackPtr++] = rightChild - nodeOffset;

                }} else if (hitLeft) {
                    nodeStack[stackPtr++] = leftChild - nodeOffset;

                } else if (hitRight) {
                    nodeStack[stackPtr++] = rightChild - nodeOffset;

                }
            }
        }
    return result;
}