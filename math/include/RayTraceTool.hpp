//
// Created by 冬榆 on 2026/1/24.
//

#ifndef RENDERINGENGINE_RAYTRACETOOL_HPP
#define RENDERINGENGINE_RAYTRACETOOL_HPP

#include <optional>
#include <random>

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)
#include "VecPro.hpp"
#else
#include "Vec.hpp"
#endif

struct Ray;
struct Material;
struct HitInfo;
struct Vertex;


struct ScatterRecord {
    Vec3 attenuation; // 衰减颜色 (f_r * cosine / pdf)
    Vec3 specularRay; // 镜面反射
    Vec3 wi;          // 采样出的入射光方向 (世界空间)
    float pdf{};      // 概率密度
};


std::optional<HitInfo> MollerTrumbore(
    const Vertex& v1,
    const Vertex& v2,
    const Vertex& v3,
    const Vec3& rayPosiModel,
    const Vec3& rayDirModel,
    const Material* material,
    float& closestT,
    uint16_t modelID);
float GetRandomFloat();
Vec4 SampleCosineHemisphere(const Vec4& N_);
inline float RandomFloat() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}
#endif //RENDERINGENGINE_RAYTRACETOOL_HPP