//
// Created by yyd on 2026/2/3.
//
#define CONSTANT_DEFINE
#include "PathTracing.cuh"
#include "Math.cuh"
#include "HitTool.cuh"
#include "Mesh.cuh"
#include "Mesh.hpp"
#include "Ray.cuh"

__global__ void pathTracing(F2PGPU* resultGPU, int SPP, int maxDepth) {
    // 获取线程ID
    const size_t width = CmDataGPu.width;
    const size_t height = CmDataGPu.height;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    size_t y = idx / width;
    size_t x = idx % width;


    // NDC坐标计算（保持不变）
    const float ndcX = 2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(width) - 1.0f;
    const float ndcY = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
    const float camX = ndcX * CmDataGPu.Asp * CmDataGPu.Scale;
    const float camY = ndcY * CmDataGPu.Scale;
    constexpr float camZ = -1.0f;

    // 射线生成
    const float4 rayDirCamera{camX, camY, camZ, 0.0f};
    const float4 rayDirWorld = normalize(CmDataGPu.cameraRot * rayDirCamera);
    const Ray ray(CmDataGPu.cameraPos, rayDirWorld);
    float3 radiance(0.0f, 0.0f, 0.0f);
    // Path Tracing（保持不变）
    for (int s = 0; s < SPP; ++s) {
        Ray currentRay = ray;
        float3 throughput(1.0f, 1.0f, 1.0f);
        for (int depth = 0; depth < maxDepth; ++depth) {
            // 这里需要处理一下求交的逻辑
            const HitInfo hitInfo = GetClosestHit(currentRay);
            if (!hitInfo.Valid) break;
            const auto material = materialsGPU[hitInfo.matID];
            float3 hitAlbedo = Sample(hitInfo.hitUV, material.KdMapId);
            float3 hitEmission = Hadamard(material.Ke, hitAlbedo);
            radiance += Hadamard(throughput, hitEmission);

            // NEE: 显式对 emissive 三角面光源采样
            float3 hitPos3 = {hitInfo.hitPos.x, hitInfo.hitPos.y, hitInfo.hitPos.z};
            float3 hitN3 = {hitInfo.hitNormal.x, hitInfo.hitNormal.y, hitInfo.hitNormal.z};
            constexpr float SHADOW_EPS = 1e-4f;

            if (emissiveTriNums > 0 && totalEmissiveArea > 0.0f) {
                float rTri = GetRandomFloatGPU() * totalEmissiveArea;
                size_t lo = 0, hi = emissiveTriNums;
                while (lo < hi) {
                    size_t mid = (lo + hi) / 2;
                    if (emissiveCDFGPU[mid] < rTri) lo = mid + 1;
                    else hi = mid;
                }
                if (lo >= emissiveTriNums) lo = emissiveTriNums - 1;
                const EmissiveTriGPU& etri = emissiveTrisGPU[lo];

                float r1 = GetRandomFloatGPU();
                float r2 = GetRandomFloatGPU();
                float sqrtR1 = sqrtf(r1);
                float u = 1.0f - sqrtR1;
                float v = r2 * sqrtR1;
                float w = 1.0f - u - v;
                float3 lightPoint = etri.v0 * u + etri.v1 * v + etri.v2 * w;

                float3 e1 = etri.v1 - etri.v0;
                float3 e2 = etri.v2 - etri.v0;
                float3 lightN = normalize(cross(e1, e2));

                float3 Lvec = lightPoint - hitPos3;
                float dist2 = dot(Lvec, Lvec);
                if (dist2 > 1e-8f) {
                    float invDist = rsqrtf(dist2);
                    float3 wi = Lvec * invDist;
                    float NdotL = dot(hitN3, wi);
                    float NdotL_light = -dot(lightN, wi);
                    if (NdotL > 0.0f && NdotL_light > 0.0f) {
                        float lightDist = 1.0f / invDist;
                        float3 shadowOrigin = hitPos3 + hitN3 * SHADOW_EPS;
                        if (!OcclusionTest(shadowOrigin, wi, lightDist - 2.0f * SHADOW_EPS)) {
                            float G = NdotL * NdotL_light / dist2;
                            float3 contrib = etri.emission * (G * totalEmissiveArea);
                            radiance += Hadamard(throughput, Hadamard(hitAlbedo, contrib));
                        }
                    }
                }
            }

            // 俄罗斯轮盘赌
            if (depth >= 3) {
                if (float maxC = max(throughput.x, max(throughput.y, throughput.z));
                    maxC < 0.1f) {
                const float q = max(0.05f, maxC);
                if (GetRandomFloatGPU() > q) break;
                throughput /= q;
                }
            }

            // BSDF
            float4 hitPos = currentRay.orignPosi + currentRay.Direction * hitInfo.t;
            float4 nextDir = SampleCosineHemisphere(hitInfo.hitNormal);
            // if (dot(nextDir, hitInfo.hitNormal) < 0.0f) break;  // 避免分支发散，去掉这个分支
            throughput = Hadamard(throughput, hitAlbedo);
            constexpr float EPS = 1e-4f;
            currentRay.orignPosi = hitPos + hitInfo.hitNormal * EPS;
            currentRay.Direction = nextDir;
        }
    }
    radiance /= static_cast<float>(SPP);
    F2PGPU pix;
    pix.x = x;
    pix.y = y;
    pix.Albedo = float4(radiance.x, radiance.y, radiance.z, 1.0f);
    pix.depth = 0.0f;
    resultGPU[idx] = pix;
}
