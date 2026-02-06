//
// Created by yyd on 2026/2/3.
//

#ifndef RENDERINGENGINE_HITTOOL_CUH
#define RENDERINGENGINE_HITTOOL_CUH
#include <cstdint>


struct SubMeshGPU;
struct MeshGPU;
struct VertexGPU;
struct BVHDataGPU;
struct Ray;
struct Mat4GPU;

struct HitInfo {
    float t{};        // 射线距离
    float4 hitPos{};      // 碰撞点坐标 (模型空间)
    float4 hitNormal{};   // 碰撞点法线 (模型空间)
    float2 hitUV{};    // 碰撞点纹理坐标
    std::uint16_t model{};   // 模型ID
    uint32_t matID{};  // 材质指针
    bool Valid;
    __device__ void trans2World(const Mat4GPU& ModelMat, const Mat4GPU& NormalWorldMat);
};

__device__ HitInfo GetClosestHit(const Ray& worldRay);
__device__ HitInfo MollerTrumbore(const VertexGPU& v1, const VertexGPU& v2, const VertexGPU& v3,
                                  const float3& rayPosiModel, const float3& rayDirModel, float& closestT);

#endif //RENDERINGENGINE_HITTOOL_CUH