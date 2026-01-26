//
// Created by 冬榆 on 2026/1/24.
//

#include "RayTraceTool.hpp"
#include "HitInfo.hpp"
#include "Mesh.hpp"
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
    const Material* material,
    float& closestT,
    const uint16_t modelID) {
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
    const float b1 = dot(S1, S) * invDet;           // u (对应 v2 的权重)
    const float b2 = dot(S2, rayDirModel) * invDet; // v (对应 v3 的权重)
    // 范围检测：t > epsilon 且 点在三角形内
    if (t > 0.001f && b1 >= 0.0f && b2 >= 0.0f && (b1 + b2) <= 1.0f) {
        // 深度检测：比当前最近的更近
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
            hit.model = modelID;
            hit.material = material;
            return hit;
        }
    }
    return std::nullopt;
}

// 获取 0-1 的随机 float
float GetRandomFloat() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

// 根据法线进行余弦加权采样
Vec4 SampleCosineHemisphere(const Vec4& N_) {
    const Vec3 N = {N_[0], N_[1], N_[2]};
    const float r1 = GetRandomFloat();
    const float r2 = GetRandomFloat();

    // 计算局部空间坐标 (Local Space)
    // theta 根据 sqrt(1-r2) 分布，偏向法线
    const float phi = 2.0f * 3.1415926f * r1;
    const float sinTheta = std::sqrt(1.0f - r2);
    const float cosTheta = std::sqrt(r2);
    const float x = sinTheta * std::cos(phi);
    const float y = sinTheta * std::sin(phi);
    const float z = cosTheta;
    // 构建 TBN 矩阵(将局部坐标转到世界坐标)
    const Vec3 up = (std::abs(N[2]) < 0.999f) ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 T = normalize(cross(up, N));
    const Vec3 B = cross(N, T);
    Vec3 result = normalize(T * x + B * y + N * z);
    // 转换并返回世界空间方向(齐次)
    return {result[0], result[1], result[2], 0.0f};
}

std::optional<HitInfo> Graphic::GetClosestHit(Ray ray,
    const std::vector<uint16_t>& models, const std::vector<RenderObjects>& renderObj) {
    std::optional<HitInfo> hitInfo;
    float closestT = std::numeric_limits<float>::max();

    for (const auto& model : models) {
        const auto& obj = renderObj.at(model);
        const auto mesh = obj.getMesh();
        Mat4 invM = obj.invModelMat();  // model->world  注意这里是立即计算，并不会更新模型自身的TRS
        Vec4 rayDirModel = invM * ray.Direction;
        Vec4 rayPosModel = invM * ray.orignPosi;
        auto rayDir = Vec3{rayDirModel[0], rayDirModel[1], rayDirModel[2]};
        auto rayPos = Vec3{rayPosModel[0], rayPosModel[1], rayPosModel[2]};
        for (const SubMesh& subMesh : *mesh) {
            // 遍历三角形
            auto start = subMesh.getOffset();
            auto end = start + subMesh.getIdxCount();
            const auto material = subMesh.getMaterial().get();
            for (auto i = start; i < end; i += 3) {
                Vertex v1 = mesh->VBO[ mesh->EBO[i] ];
                Vertex v2 = mesh->VBO[ mesh->EBO[i+1] ];
                Vertex v3 = mesh->VBO[ mesh->EBO[i+2] ];
                // 碰撞检测
                auto hit = MollerTrumbore(v1, v2, v3,
                    rayPos, rayDir, material, closestT, model);
                if (!hit) continue;
                hitInfo = hit;
            }
        }
    }
    if (!hitInfo) return std::nullopt;
    auto& model = renderObj.at(hitInfo->model);
    auto& M = model.ModelMatUnsafe();  // 为了并发性，这里不保证模型更新，需要在在外部保证更新
    auto& NM = model.NormalMatUnsafe();
    hitInfo->trans2World(M, NM);
    return hitInfo;
}