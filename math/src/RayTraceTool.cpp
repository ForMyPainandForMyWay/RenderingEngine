//
// Created by 冬榆 on 2026/1/24.
//

#include "RayTraceTool.hpp"
#include "BVH.hpp"
#include "Engine.hpp"
#include "HitInfo.hpp"
#include "Ray.hpp"
#include "Shape.hpp"
#include "RenderObjects.hpp"
#include "Graphic.hpp"


std::optional<HitInfo> MollerTrumbore(
    const Vertex& v1,
    const Vertex& v2,
    const Vertex& v3,
    const Vec3& rayPosiModel,
    const Vec3& rayDirModel,
    float& closestT) {
    const Vec3 p0 = v1.position;
    const Vec3 p1 = v2.position;
    const Vec3 p2 = v3.position;
    const Vec3 E1 = p1 - p0;
    const Vec3 E2 = p2 - p0;
    const Vec3 S = rayPosiModel - p0;
    const Vec3 S1 = cross(rayDirModel, E2);
    const Vec3 S2 = cross(S, E1);
    const float det = dot(S1, E1);
    // 平行检测 (det=0)
    if (std::abs(det) < 1e-6f) return std::nullopt;
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
            hit.t = t;
            // 重心坐标插值
            // b1 是 v2 的权重，b2 是 v3 的权重，剩下的 w 是 v1 的权重
            const float w = 1.0f - b1 - b2;
            Vec3 hitPos = rayPosiModel + rayDirModel * t;
            hit.hitPos = {hitPos[0], hitPos[1], hitPos[2], 1.0f};  // 齐次坐标
            Vec3 hitNormal = normalize(v1.normal * w + v2.normal * b1 + v3.normal * b2);
            hit.hitNormal = {hitNormal[0], hitNormal[1], hitNormal[2], 0.0f};  // 齐次坐标
            hit.hitUV = v1.uv * w + v2.uv * b1 + v3.uv * b2;
            return hit;
        }
    }
    return std::nullopt;
}

float GetRandomFloat() {
    static std::mt19937 gen([] {
        std::random_device rd;
        return rd();
    }());
    static std::uniform_real_distribution dis(0.0f, 1.0f);
    return dis(gen);
}

// 根据法线进行余弦加权采样
Vec4 SampleCosineHemisphere(const Vec4& N_) {
    const Vec3 N = {N_[0], N_[1], N_[2]};
    const float r1 = GetRandomFloat();
    const float r2 = GetRandomFloat();

    // 计算局部空间坐标
    // theta 根据 sqrt(1-r2) 分布，偏向法线
    const float phi = 2.0f * 3.1415926f * r1;
    const float sinTheta = std::sqrt(1.0f - r2);
    const float cosTheta = std::sqrt(r2);
    const float x = sinTheta * std::cos(phi);
    const float y = sinTheta * std::sin(phi);
    const float z = cosTheta;
    // 构建 TBN 矩阵(将局部坐标转到世界坐标)
    const Vec3 up = std::abs(N[2]) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 T = normalize(cross(up, N));
    const Vec3 B = cross(N, T);
    Vec3 result = normalize(T * x + B * y + N * z);
    // 转换并返回世界空间方向(齐次)
    return {result[0], result[1], result[2], 0.0f};
}

std::optional<HitInfo> Engine::GetClosestHit(const Ray& worldRay) const {
    std::optional<HitInfo> closestHit = std::nullopt;
    float closestT = std::numeric_limits<float>::max(); // 全局最近距离
    if (tlas->nodes.empty()) return std::nullopt;
    //  TLAS 遍历栈
    const Vec3 rayDirInv = { 1.0f / worldRay.Direction[0], 1.0f / worldRay.Direction[1], 1.0f / worldRay.Direction[2] };
    const Vec3 rayOrigin = { worldRay.orignPosi[0], worldRay.orignPosi[1], worldRay.orignPosi[2] };
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // 压入根节点
    // 遍历 TLAS
    while (stackPtr > 0) {
        const int nodeIdx = stack[--stackPtr];
        const auto&[bbox, leftChild, rightChild, contentRef, isLeaf] = tlas->nodes[nodeIdx];
        // TLAS 节点 AABB 测试
        if (float tEntry;
            !IntersectAABB(bbox, rayOrigin, rayDirInv, closestT, tEntry)) {
            continue;
        }
        // 如果是叶子节点 (命中 Instance)
        if (isLeaf) {
            const Instance& inst = tlas->instances[contentRef];
            // 将射线变换到模型空间
            Ray localRay = TransformRayToModel(worldRay, inst.invTransform);
            // 获取对应的 BLAS
            const auto blas = blasList[inst.blasIdx];
            // 遍历 BLAS
            // D. 如果命中，处理结果
            if (auto subHit = blas->Intersect(localRay, closestT);
                subHit.has_value()) {
                // 更新全局最近距离
                closestT = subHit->t;
                // 构造世界空间的 HitInfo
                HitInfo worldHit;
                worldHit.t = closestT;
                // P = O + t * D
                worldHit.hitPos = worldRay.orignPosi + worldRay.Direction * closestT;
                Mat4 invTrans = inst.invTransform.Transpose();
                Vec4 localN = {subHit->hitNormal[0], subHit->hitNormal[1], subHit->hitNormal[2], 0.0f}; // w=0
                Vec4 worldN = invTrans * localN;
                worldHit.hitNormal = normalize({worldN[0], worldN[1], worldN[2]});
                worldHit.mat = subHit->mat; // 材质传递
                worldHit.hitUV = subHit->hitUV;
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