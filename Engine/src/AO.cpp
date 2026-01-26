//
// Created by 冬榆 on 2026/1/19.
//

#include "Engine.hpp"
#include "Graphic.hpp"

#include <vector>
#include <cmath>
#include <random>

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
    // 临时存储原始 AO
    std::vector<FloatPixel> rawAO(outBuffer.size());
    // ---------- Step 1: 计算原始 AO ----------
    for (auto y = 0; y < h; y++) {
        for (auto x = 0; x < w; x++) {
            const auto idx = y * w + x;
            const Vec4& worldPos = Gdata[idx].worldPosi; // 从Gdata中获取世界坐标
            auto viewPos = ViewMat * worldPos;
            auto& worldnormal = Gdata[idx].normal; // 从Gdata中获取法线
            auto viewnormaltmp = ViewMat * Vec4{worldnormal[0], worldnormal[1], worldnormal[2], 0};
            Vec3 normal = normalize(Vec3{viewnormaltmp[0], viewnormaltmp[1], viewnormaltmp[2]});

            // 构建 TBN
            Vec3 randvec = normalize(randomNoiseVector());
            Vec3 tangent = normalize(randvec - normal * dot(randvec, normal));
            Mat3 TBN = Mat3(tangent, cross(normal, tangent), normal).Transpose();
            int count = 0;
            for (int i = 0; i < sampleCount; i++) {
                Vec3 randomVec = TBN * randomUnitHemisphere();
                Vec4 tmp = {randomVec[0], randomVec[1], randomVec[2], 0.f};
                const Vec4 randomPos = viewPos + tmp * radius;
                Vec4 rclipPos = Projection * randomPos;
                Vec4 rndcPos = rclipPos / rclipPos[3];
                const int sx = static_cast<int>((rndcPos[0] * 0.5f + 0.5f) * static_cast<float>(w));
                const int sy = static_cast<int>((1.0f - (rndcPos[1] * 0.5f + 0.5f)) * static_cast<float>(h));
                if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
                const float ndcDepth = depthBuffer[sx + sy * w];
                if (const float sampleViewDepth = rndcPos[2]; ndcDepth < sampleViewDepth - 3e-3f)
                    count++;
            }
            const float ao = 1.0f - (static_cast<float>(count) / static_cast<float>(sampleCount));
            rawAO[idx].r = ao;
            rawAO[idx].g = ao;
            rawAO[idx].b = ao;
        }
    }

    // ---------- Step 2: 双边模糊 ----------
    // 参数，可根据需要调整
    constexpr float sigma_s = 2.0f;
    auto SpatialWeight = [&](const int dx, const int dy) {
        const auto dist2 = static_cast<float>(dx * dx + dy * dy);
        return expf(-dist2 / (2.0f * sigma_s * sigma_s));
    };
    const std::vector<FloatPixel> temp = rawAO;
    for (auto y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            constexpr int radiusBlur = 2;
            FloatPixel sum{0,0,0};
            float wsum = 0.0f;
            const float centerDepth = depthBuffer[y*w + x];
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
            outBuffer[idx].r = sum.r / wsum * inBuffer[idx].r;
            outBuffer[idx].g = sum.g / wsum * inBuffer[idx].g;
            outBuffer[idx].b = sum.b / wsum * inBuffer[idx].b;
        }
    }
}