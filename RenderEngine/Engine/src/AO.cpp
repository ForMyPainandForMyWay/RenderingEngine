//
// Created by 冬榆 on 2026/1/19.
//

#include "Engine.hpp"
#include "Graphic.hpp"

#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
#include <thread>

// 生成单位半球随机向量
inline Vec3 randomUnitHemisphere() {
     static std::mt19937 gen(12345);
     static std::uniform_real_distribution dis(0.0f, 1.0f);

     const float phi = 2.0f * 3.14159265359f * dis(gen);
     float cosTheta = dis(gen);
     const float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);

     return {std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta};
}

// 生成随机旋转噪声向量 (XY 平面)
Vec3 randomNoiseVector() {
    static std::mt19937 gen(54321);
    static std::uniform_real_distribution dis(0.0f, 1.0f);
    const float angle = dis(gen) * 2.0f * 3.14159265359f; // 随机角度
    return Vec3{std::cos(angle), std::sin(angle), 0.0f}; // 单位圆上的均匀方向
}

float random_float(const float min, const float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution dis(min, max);
    return dis(gen);
}


void Graphic::SSAO(
    const std::vector<FloatPixel> &inBuffer,
    std::vector<FloatPixel> &outBuffer,
    const std::vector<GBufferData> &Gdata,
    const std::vector<float> &depthBuffer,  // NDC空间
    const Mat4 &ViewMat,
    const Mat4 &Projection,
    const float radius,
    const int sampleCount) const {
    const auto w = engine->width;
    const auto h = engine->height;
    const auto fw = static_cast<float>(w);
    const auto fh = static_cast<float>(h);
    const float p00 = Projection[0][0];
    const float p11 = Projection[1][1];
    const float p22 = Projection[2][2];
    const float p23 = Projection[2][3];
    constexpr float backgroundDepth = 0.9999f;
    constexpr float depthBias = 0.01f;

    auto reconstructViewPos = [&](const int px, const int py, const float ndcZ) {
        const float ndcX = (2.0f * (static_cast<float>(px) + 0.5f) / fw) - 1.0f;
        const float ndcY = 1.0f - (2.0f * (static_cast<float>(py) + 0.5f) / fh);
        const float viewZ = -p23 / (ndcZ + p22);
        const float viewX = ndcX * (-viewZ) / p00;
        const float viewY = ndcY * (-viewZ) / p11;
        return Vec3{viewX, viewY, viewZ};
    };

    std::vector<FloatPixel> rawAO(outBuffer.size());
    
    const auto threadCount = std::thread::hardware_concurrency();
    const int chunkSize = std::max(1, (static_cast<int>(h) + static_cast<int>(threadCount) - 1) / static_cast<int>(threadCount));
    std::vector<std::future<void>> rawAOFutures;
    
    for (int startY = 0; startY < h; startY += chunkSize) {
        const int endY = std::min(startY + chunkSize, static_cast<int>(h));
        rawAOFutures.emplace_back(engine->pool.addTask([&, startY, endY]() {
            for (auto y = startY; y < endY; y++) {
                for (auto x = 0; x < w; x++) {
                    const auto idx = y * w + x;
                    const float ndcDepth = depthBuffer[idx];
                    if (ndcDepth >= backgroundDepth) {
                        rawAO[idx].r = 1.0f;
                        rawAO[idx].g = 1.0f;
                        rawAO[idx].b = 1.0f;
                        rawAO[idx].i = 0.0f;
                        continue;
                    }

                    const Vec3 viewPos = reconstructViewPos(x, y, ndcDepth);
                    auto& worldnormal = Gdata[idx].normal;
                    auto viewnormaltmp = ViewMat * Vec4{worldnormal[0], worldnormal[1], worldnormal[2], 0};
                    Vec3 normal = normalize(Vec3{viewnormaltmp[0], viewnormaltmp[1], viewnormaltmp[2]});
                    if (dot(normal, normal) < 1e-6f) {
                        rawAO[idx].r = 1.0f;
                        rawAO[idx].g = 1.0f;
                        rawAO[idx].b = 1.0f;
                        rawAO[idx].i = 0.0f;
                        continue;
                    }

                    Vec3 randvec = normalize(randomNoiseVector());
                    Vec3 tangent = randvec - normal * dot(randvec, normal);
                    if (dot(tangent, tangent) < 1e-6f) {
                        tangent = std::abs(normal[2]) < 0.999f
                            ? normalize(cross(Vec3{0.0f, 0.0f, 1.0f}, normal))
                            : normalize(cross(Vec3{0.0f, 1.0f, 0.0f}, normal));
                    } else {
                        tangent = normalize(tangent);
                    }
                    Vec3 bitangent = normalize(cross(normal, tangent));
                    Mat3 TBN = Mat3(tangent, bitangent, normal).Transpose();
                    float occlusion = 0.0f;
                    for (int i = 0; i < sampleCount; i++) {
                        Vec3 randomVec = TBN * randomUnitHemisphere();
                        const Vec3 samplePos = viewPos + randomVec * radius;
                        Vec4 rclipPos = Projection * Vec4{samplePos[0], samplePos[1], samplePos[2], 1.0f};
                        if (std::abs(rclipPos[3]) < 1e-6f) continue;
                        Vec4 rndcPos = rclipPos / rclipPos[3];
                        const int sx = static_cast<int>((rndcPos[0] * 0.5f + 0.5f) * fw);
                        const int sy = static_cast<int>((1.0f - (rndcPos[1] * 0.5f + 0.5f)) * fh);
                        if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
                        const float sampleDepth = depthBuffer[sx + sy * w];
                        if (sampleDepth >= backgroundDepth) continue;

                        const Vec3 geoViewPos = reconstructViewPos(sx, sy, sampleDepth);
                        if (geoViewPos[2] < samplePos[2] + depthBias) continue;

                        const Vec3 delta = geoViewPos - viewPos;
                        const float dist = std::sqrt(dot(delta, delta));
                        if (dist >= radius) continue;

                        const float rangeCheck = 1.0f - dist / radius;
                        occlusion += rangeCheck;
                    }
                    const float ao = 1.0f - (occlusion / static_cast<float>(sampleCount));
                    rawAO[idx].r = ao;
                    rawAO[idx].g = ao;
                    rawAO[idx].b = ao;
                    rawAO[idx].i = 0.0f;
                }
            }
        }));
    }
    
    for (auto& future : rawAOFutures) {
        future.wait();
    }

    // 双边模糊
    constexpr float sigma_s = 2.0f;
    auto SpatialWeight = [&](const int dx, const int dy) {
        const auto dist2 = static_cast<float>(dx * dx + dy * dy);
        return expf(-dist2 / (2.0f * sigma_s * sigma_s));
    };
    const std::vector<FloatPixel> temp = rawAO;
    
    std::vector<std::future<void>> blurFutures;
    for (int startY = 0; startY < h; startY += chunkSize) {
        const int endY = std::min(startY + chunkSize, static_cast<int>(h));
        blurFutures.emplace_back(engine->pool.addTask([&, startY, endY]() {
            for (auto y = startY; y < endY; y++) {
                for (int x = 0; x < w; x++) {
                    constexpr int radiusBlur = 2;
                    FloatPixel sum{0,0,0};
                    float wsum = 0.0f;
                    const float centerDepth = depthBuffer[y*w + x];
                    if (centerDepth >= backgroundDepth) {
                        const auto idx = y * w + x;
                        outBuffer[idx] = inBuffer[idx];
                        continue;
                    }
                    for (int dy = -radiusBlur; dy <= radiusBlur; dy++) {
                        const int ny = y + dy;
                        if (ny < 0 || ny >= h) continue;
                        for (int dx = -radiusBlur; dx <= radiusBlur; dx++) {
                            constexpr float sigma_r = 0.1f;
                            const int nx = x + dx;
                            if (nx < 0 || nx >= w) continue;
                            const auto idx = ny * w + nx;
                            const float ws = SpatialWeight(dx, dy);
                            const float wr = expf(-powf(depthBuffer[idx] - centerDepth, 2) / (2.0f * sigma_r * sigma_r));
                            const float w1 = ws * wr;
                            sum.r += temp[idx].r * w1;
                            sum.g += temp[idx].g * w1;
                            sum.b += temp[idx].b * w1;
                            wsum += w1;
                        }
                    }
                    const auto idx = y * w + x;
                    const float ao = wsum > 1e-6f ? std::clamp(sum.r / wsum, 0.0f, 1.0f) : 1.0f;
                    constexpr float aoExponent = 2.0f;
                    constexpr float aoStrength = 1.35f;
                    const float boostedAO = std::pow(ao, aoExponent);
                    const float shadedAO = std::clamp(1.0f - aoStrength * (1.0f - boostedAO), 0.0f, 1.0f);
                    outBuffer[idx].r = shadedAO * inBuffer[idx].r;
                    outBuffer[idx].g = shadedAO * inBuffer[idx].g;
                    outBuffer[idx].b = shadedAO * inBuffer[idx].b;
                    outBuffer[idx].i = inBuffer[idx].i;
                }
            }
        }));
    }
    
    for (auto& future : blurFutures) {
        future.wait();
    }
}
