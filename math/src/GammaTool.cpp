//
// Created by 冬榆 on 2026/1/19.
//

#include <algorithm>
#include <cmath>

#include "GammaTool.hpp"

float srgbToLinear(const float c) {
    if (c <= 0.04045f)
        return c / 12.92f;
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

float linearToSrgb(float c) {
    c = std::clamp(c, 0.0f, 1.0f);
    if (c <= 0.0031308f)
        return 12.92f * c;
    return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}
