//
// Created by 冬榆 on 2026/1/3.
//

#include "Mesh.hpp"
#include "FragTool.hpp"
#include "LerpTool.hpp"

// 贴图采样
// 未插值
FloatPixel Sample(const VecN<2>& uv, const std::shared_ptr<TextureMap>& texture) {
    if (texture == nullptr) return {1.0f, 1.0f, 1.0f}; // 纹理不存在时返回白色
    const auto width = static_cast<float>(texture->width);
    const auto height = static_cast<float>(texture->height);
    // 确保 uv 在 [0, 1] 范围内
    const float u = std::clamp(uv[0], 0.0f, 1.0f);
    const float v = std::clamp(uv[1], 0.0f, 1.0f);

    // 计算实际的纹理坐标
    const float x = u * (width - 1);
    const float y = v * (height - 1);
    const auto x0 = static_cast<size_t>(std::floor(x));
    const auto y0 = static_cast<size_t>(std::floor(y));
    return texture->uvImg->floatImg[y0 * texture->width + x0];
}

// 双线性插值
FloatPixel BilinearSample(const VecN<2>& uv, const std::shared_ptr<TextureMap>& texture) {
    if (texture == nullptr) return {1.0, 1.0, 1.0};
    const auto width = static_cast<float>(texture->width);
    const auto height = static_cast<float>(texture->height);

    // 确保 uv 在 [0, 1] 范围内
    const float u = std::clamp(uv[0], 0.0f, 1.0f);
    const float v = std::clamp(uv[1], 0.0f, 1.0f);

    // 计算实际的纹理坐标
    const float x = u * (width - 1);
    const float y = v * (height - 1);

    // 获取最近的四个整数坐标
    const auto x0 = static_cast<size_t>(std::floor(x));
    const size_t x1 = std::min(x0 + 1, static_cast<size_t>(width - 1));
    const auto y0 = static_cast<size_t>(std::floor(y));
    const size_t y1 = std::min(y0 + 1, static_cast<size_t>(height - 1));

    // 计算插值系数
    const float sx = x - static_cast<float>(x0);
    const float sy = y - static_cast<float>(y0);

    // 获取四个邻近像素
    const auto& floatImg = texture->uvImg->floatImg;
    const FloatPixel c00 = floatImg[y0 * texture->width + x0];
    const FloatPixel c10 = floatImg[y0 * texture->width + x1];
    const FloatPixel c01 = floatImg[y1 * texture->width + x0];
    const FloatPixel c11 = floatImg[y1 * texture->width + x1];

    // 在 x 方向上插值
    const FloatPixel c0 = lerp(c00, c10, sx);
    const FloatPixel c1 = lerp(c01, c11, sx);

    // 在 y 方向上插值
    return lerp(c0, c1, sy);
}