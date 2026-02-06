//
// Created by yyd on 2026/2/3.
//
#include <cfloat>

#include "HitTool.cuh"
#include "Math.cuh"
#include "PathTracing.cuh"
#include "Ray.cuh"

__device__ void HitInfo::trans2World(const Mat4GPU& ModelMat, const Mat4GPU& NormalWorldMat){
    hitPos = ModelMat * hitPos;
    hitPos = hitPos / hitPos.z;
    hitNormal = normalize(NormalWorldMat * hitNormal);
}

__device__ HitInfo GetClosestHit(const Ray& worldRay) {
    HitInfo closestHit;
    closestHit.Valid = false;
    float closestT = FLT_MAX; // 全局最近距离
    if (tlasNodeNums == 0) return closestHit;
    //  TLAS 遍历栈
    const float3 rayDirInv = { 1.0f / worldRay.Direction.x, 1.0f / worldRay.Direction.y, 1.0f / worldRay.Direction.z };
    const float3 rayOrigin = { worldRay.orignPosi.x, worldRay.orignPosi.y, worldRay.orignPosi.z };
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // 压入根节点
    // 遍历 TLAS
    while (stackPtr > 0) {
        const int nodeIdx = stack[--stackPtr];
        const auto&[bbox, leftChild, rightChild, contentRef, isLeaf] = TlasNodesGPU[nodeIdx];
        // TLAS 节点 AABB 测试
        if (float tEntry;
            !IntersectAABBGPU(bbox, rayOrigin, rayDirInv, closestT, tEntry)) {
            continue;
        }
        // 如果是叶子节点 (命中 Instance)
        if (isLeaf) {
            const InstanceGPU& inst = TlasInstanceGPU[contentRef];
            // 将射线变换到模型空间
            Ray localRay = TransformRayToModel(worldRay, inst.invTransform);
            // 获取对应的 BLAS
            const auto& blas = blasGPU[inst.blasIdx];
            auto subHit = blas.Intersect(localRay, closestT);
            // 遍历 BLAS
            if (subHit.Valid) {
                // 更新全局最近距离
                closestT = subHit.t;
                // 构造世界空间的 HitInfo
                HitInfo worldHit;
                worldHit.Valid = true;
                worldHit.t = closestT;
                // P = O + t * D
                worldHit.hitPos = worldRay.orignPosi + worldRay.Direction * closestT;
                Mat4GPU invTrans = Transpose4(inst.invTransform);
                float4 localN = {subHit.hitNormal.x, subHit.hitNormal.y, subHit.hitNormal.z, 0.0f}; // w=0
                float4 worldN = invTrans * localN;
                worldHit.hitNormal = normalize({worldN.x, worldN.y, worldN.z, 0.0f});
                worldHit.matID = subHit.matID; // 材质传递
                worldHit.hitUV = subHit.hitUV;
                closestHit = worldHit;
            }
        }
        // 子节点入栈
        else {
            stack[stackPtr++] = rightChild;
            stack[stackPtr++] = leftChild;
        }
    }
    return closestHit;
}

__device__ HitInfo MollerTrumbore(const VertexGPU& v1, const VertexGPU& v2, const VertexGPU& v3,
                                  const float3& rayPosiModel, const float3& rayDirModel, float& closestT) {

    const float3 p0 = v1.position;
    const float3 p1 = v2.position;
    const float3 p2 = v3.position;
    const float3 E1 = p1 - p0;
    const float3 E2 = p2 - p0;
    const float3 S = rayPosiModel - p0;
    const float3 S1 = cross(rayDirModel, E2);
    const float3 S2 = cross(S, E1);
    const float det = dot(S1, E1);
    HitInfo hitFail{};
    hitFail.Valid = false;
    // 平行检测 (det=0)
    if (std::abs(det) < 1e-6f) return hitFail;
    const float invDet = 1.0f / det;
    const float t = dot(S2, E2) * invDet;
    const float b1 = dot(S1, S) * invDet;  // u (对应 v2 的权重)
    // t > epsilon 且 点在三角形内
    if (const float b2 = dot(S2, rayDirModel) * invDet;
        t > 0.001f && b1 >= 0.0f && b2 >= 0.0f && b1 + b2 <= 1.0f) {
        // 深度检测
        if (t < closestT) {
            closestT = t; // 更新最近距离
            HitInfo hit;
            hit.Valid = true;
            hit.t = t;
            // 重心坐标插值
            // b1 是 v2 的权重，b2 是 v3 的权重，剩下的 w 是 v1 的权重
            const float w = 1.0f - b1 - b2;
            const auto [x, y, z] = rayPosiModel + rayDirModel * t;
            hit.hitPos = {x, y, z, 1.0f};  // 齐次坐标
            const auto [x_n, y_n, z_n] = normalize(v1.normal * w + v2.normal * b1 + v3.normal * b2);
            hit.hitNormal = {x_n, y_n, z_n, 0.0f};  // 齐次坐标
            hit.hitUV = v1.texCoord * w + v2.texCoord * b1 + v3.texCoord * b2;
            return hit;
        }}
    return hitFail;
}