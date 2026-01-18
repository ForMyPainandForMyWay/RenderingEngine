//
// Created by 冬榆 on 2026/1/9.
//

#include <fstream>

#include "ShadowMap.h"


ShadowMap::ShadowMap(const size_t width, const size_t height) {
    ZBufferShadow.resize(width * height, 1.0f);
    this->width = width;
    this->height = height;
}

float &ShadowMap::operator[](const size_t i) {
    return ZBufferShadow[i];
}

const float &ShadowMap::operator[](const size_t i) const {
    return ZBufferShadow[i];
}

void ShadowMap::clear() {
    std::ranges::fill(ZBufferShadow, 1.0f);
}

void ShadowMap::resize(const size_t w, const size_t h) {
    ZBufferShadow.resize(w * h);
    this->width = w;
    this->height = h;
    clear();
}

float ShadowMap::Sample(const float u, const float v) const {
    const float fx = u * (static_cast<float>(width) - 1.0f);
    const float fy = v * (static_cast<float>(height) - 1.0f);
    int x = static_cast<int>(std::lroundf(fx));
    int y = static_cast<int>(std::lroundf(fy));
    x = std::clamp(x, 0, static_cast<int>(width) - 1);
    y = std::clamp(y, 0, static_cast<int>(height) - 1);
    return ZBufferShadow[y * width + x];
}

float ShadowMap::SamplePCF(
    const float currentDepth,
    const float bias,
    const float u,
    const float v,
    const int R) const {
    const float fx = u * static_cast<float>(width - 1);
    const float fy = v * static_cast<float>(height - 1);
    const auto cx = lroundf(fx);
    const auto cy = lroundf(fy);
    float shadow = 0.0f;
    int count = 0;
    for (auto dy = -R; dy <= R; ++dy)
        for (auto dx = -R; dx <= R; ++dx) {
            const auto x = cx + dx;
            const auto y = cy + dy;
            if (x < 0 || x >= width || y < 0 || y >= height)
                continue;
            if (const float depth=ZBufferShadow[y*width+x]; currentDepth-bias>depth)
                shadow += 1.0f;
            count++;
        }
    if (count == 0) return 1.0f; // 当作完全照亮
    return 1 - shadow / static_cast<float>(count); // 0 = 亮，1 = 阴影
}

float ShadowMap::SamplePCSS(
    const float currentDepth,
    const float bias,
    const float u,
    const float v,
    const float lightSizeUV,
    const int blockerSearchR,
    const int minPCFR,
    const int maxPCFR) const {
    const float fx = u * (static_cast<float>(width)  - 1);
    const float fy = v * (static_cast<float>(height) - 1);
    const int cx = static_cast<int>(roundf(fx));
    const int cy = static_cast<int>(roundf(fy));
    float avgBlockerDepth = 0.0f;
    if (float blockerCount = 0; !FindBlocker(
            currentDepth, bias,
            cx, cy,
            blockerSearchR,
            avgBlockerDepth,
            blockerCount)) {
        return 1.0f;}
    // 标准 PCSS 公式
    const float penumbra = (currentDepth - avgBlockerDepth) / avgBlockerDepth;
    // 转成 PCF 半径（像素）
    float filterR = penumbra * lightSizeUV * static_cast<float>(width);
    filterR = std::clamp(
        filterR,
        static_cast<float>(minPCFR),
        static_cast<float>(maxPCFR));
    const int R = static_cast<int>(std::ceil(filterR));
    // PCF
    float shadow = 0.0f;
    float count = 0;
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
            const auto x = cx + dx;
            const auto y = cy + dy;
            if (x < 0 || x >= width || y < 0 || y >= height) continue;
            if (const float depth=ZBufferShadow[y*width+x]; currentDepth-bias > depth)
                shadow += 1.0f;
            count++;
        }
    if (count == 0) return 1.0f;
    return 1.0f - shadow / count;
}


bool ShadowMap::FindBlocker(
    const float currentDepth,
    const float bias,
    const int cx,
    const int cy,
    const int searchR,
    float& avgBlockerDepth,
    float& blockerCount) const {
    float sumDepth = 0.0f;
    blockerCount = 0;
    for (int dy = -searchR; dy <= searchR; ++dy)
        for (int dx = -searchR; dx <= searchR; ++dx) {
            const int x = cx + dx;
            const int y = cy + dy;
            if (x < 0 || x >= width || y < 0 || y >= height) continue;
            if (const float depth = ZBufferShadow[y*width+x]; currentDepth-bias > depth) {
                sumDepth += depth;
                blockerCount++;
            }
        }
    if (std::abs(blockerCount) < 1e-5) return false;
    avgBlockerDepth = sumDepth / blockerCount;
    return true;
}


void ShadowMap::save() const{
    std::ofstream file("shadowmap.pgm", std::ios::binary);
    if (!file) return;

    // PGM header
    file << "P5\n";
    file << width << " " << height << "\n";
    file << "255\n";

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            float depth = ZBufferShadow[y * width + x];
            depth = std::clamp(depth, 0.0f, 1.0f);
            auto gray =
                static_cast<unsigned char>((depth) * 255.0f);

            file.write(reinterpret_cast<char*>(&gray), 1);
        }
    }
}

