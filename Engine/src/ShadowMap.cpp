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
            unsigned char gray =
                static_cast<unsigned char>((depth) * 255.0f);

            file.write(reinterpret_cast<char*>(&gray), 1);
        }
    }
}

