//
// Created by 冬榆 on 2026/1/19.
//

#include "Engine.hpp"
#include "Graphic.hpp"

constexpr float FXAA_ABSOLUTE_LUMA_THRESHOLD = 0.0312f;
constexpr float FXAA_RELATIVE_LUMA_THRESHOLD = 0.125f;
constexpr float FXAA_SUBPIX_SHIFT = 0.5f;

constexpr float FXAA_QUALITY_EDGE_THRESHOLD = 0.125f;
constexpr float FXAA_QUALITY_EDGE_THRESHOLD_MIN = 0.0312f;
constexpr int FXAA_QUALITY_SEARCH_STEPS = 8;
constexpr float FXAA_QUALITY_SUBPIX_CAP = 0.75f;

constexpr float FXAA_CONSOLE_EDGE_THRESHOLD = 0.166f;
constexpr float FXAA_CONSOLE_EDGE_THRESHOLD_MIN = 0.0625f;
constexpr int   FXAA_CONSOLE_SEARCH_STEPS = 2;
constexpr float FXAA_CONSOLE_SUBPIX_CAP   = 0.5f;


// FXAA
void Graphic::FXAA(std::vector<FloatPixel>& inBuffer, std::vector<FloatPixel>& outBuffer) const {
    const auto w = engine->width;
    const auto h = engine->height;
    // 1. 预计算亮度
    for (auto& [r, g, b, i] : inBuffer) {
        i = 0.213f * r + 0.715f * g + 0.072f * b;
    }
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const size_t cordM = x + y * w;
            const size_t cordN = cordM - w;
            const size_t cordS = cordM + w;
            const size_t cordW = cordM - 1;
            const size_t cordE = cordM + 1;

            const float lumaM = inBuffer[cordM].i;
            const float lumaN = inBuffer[cordN].i;
            const float lumaS = inBuffer[cordS].i;
            const float lumaW = inBuffer[cordW].i;
            const float lumaE = inBuffer[cordE].i;

            const float lumaMin = std::min({lumaM, lumaN, lumaS, lumaW, lumaE});
            const float lumaMax = std::max({lumaM, lumaN, lumaS, lumaW, lumaE});
            const float lumaContrast = lumaMax - lumaMin;

            // Edge threshold
            const float edgeThreshold = std::max(
                FXAA_ABSOLUTE_LUMA_THRESHOLD,
                lumaMax * FXAA_RELATIVE_LUMA_THRESHOLD
            );

            if (lumaContrast < edgeThreshold) {
                outBuffer[cordM] =inBuffer[cordM];
                continue;
            }

            // 梯度方向判断
            const float gradH = std::abs(lumaW - lumaE);
            const float gradV = std::abs(lumaN - lumaS);
            const bool isHorizontal = gradH >= gradV;

            // 垂直于边缘方向采样
            size_t offset1, offset2;
            if (isHorizontal) {
                offset1 = -w; // up
                offset2 =  w; // down
            } else {
                offset1 = -1; // left
                offset2 =  1; // right
            }

            const size_t cordA = cordM + offset1;
            const size_t cordB = cordM + offset2;
            // 这里需要的是rgb，读取inBuffer
            const auto& [cMr, cMg, cMb, _1] = inBuffer[cordM];
            const auto& [cAr, cAg, cAb, _2] = inBuffer[cordA];
            const auto& [cBr, cBg, cBb, _3] = inBuffer[cordB];

            // Sub-pixel AA 混合系数
            const float lumaAvg =
                (lumaN + lumaS + lumaW + lumaE) * 0.25f;

            const float lumaDeltaML =
                std::abs(lumaAvg - lumaM);

            float blend = 0.0f;
            if (lumaContrast > 1e-5f)
                blend = lumaDeltaML / lumaContrast;

            blend = std::clamp(blend, 0.0f, 1.0f);

            // 非线性压缩
            blend = blend * blend;
            // 上限，防止过度模糊
            blend = std::min(blend, 0.75f);

            // 沿边缘方向的过滤颜色
            const float fr = (cAr + cBr) * 0.5f;
            const float fg = (cAg + cBg) * 0.5f;
            const float fb = (cAb + cBb) * 0.5f;

            FloatPixel out{};
            out.r = cMr * (1.0f - blend) + fr * blend;
            out.g = cMg * (1.0f - blend) + fg * blend;
            out.b = cMb * (1.0f - blend) + fb * blend;

            outBuffer[cordM] = out;
        }
    }
}

void Graphic::FXAAQ(std::vector<FloatPixel>& inBuffer,
                   std::vector<FloatPixel>& outBuffer) const {
    const int w = static_cast<int>(engine->width);
    const int h = static_cast<int>(engine->height);

    // 1. 预计算亮度
    for (auto& [r, g, b, i] : inBuffer) {
        i = 0.213f * r + 0.715f * g + 0.072f * b;
    }

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {

            const size_t cordM = x + y * w;
            const size_t cordN = cordM - w;
            const size_t cordS = cordM + w;
            const size_t cordW = cordM - 1;
            const size_t cordE = cordM + 1;

            const float lumaM = inBuffer[cordM].i;
            const float lumaN = inBuffer[cordN].i;
            const float lumaS = inBuffer[cordS].i;
            const float lumaW = inBuffer[cordW].i;
            const float lumaE = inBuffer[cordE].i;

            const float lumaMin = std::min({lumaM, lumaN, lumaS, lumaW, lumaE});
            const float lumaMax = std::max({lumaM, lumaN, lumaS, lumaW, lumaE});
            const float lumaContrast = lumaMax - lumaMin;

            const float edgeThreshold = std::max(
                FXAA_QUALITY_EDGE_THRESHOLD_MIN,
                lumaMax * FXAA_QUALITY_EDGE_THRESHOLD
            );

            if (lumaContrast < edgeThreshold) {
                outBuffer[cordM] = inBuffer[cordM];
                continue;
            }

            // --------------------------------------------------
            // Edge orientation
            // --------------------------------------------------
            const float gradH = std::abs(lumaW - lumaE);
            const float gradV = std::abs(lumaN - lumaS);
            const bool isHorizontal = gradH >= gradV;

            const int step = isHorizontal ? w : 1;

            // --------------------------------------------------
            // Edge search (both directions)
            // --------------------------------------------------
            int distNeg = 0, distPos = 0;

            // negative direction
            for (int i = 1; i <= FXAA_QUALITY_SEARCH_STEPS; ++i) {
                const size_t c = cordM - step * i;
                if (c >= inBuffer.size()) break;

                if (std::abs(inBuffer[c].i - lumaM) >= lumaContrast * 0.5f) {
                    distNeg = i;
                    break;
                }
            }

            // positive direction
            for (int i = 1; i <= FXAA_QUALITY_SEARCH_STEPS; ++i) {
                const size_t c = cordM + step * i;
                if (c >= inBuffer.size()) break;

                if (std::abs(inBuffer[c].i - lumaM) >= lumaContrast * 0.5f) {
                    distPos = i;
                    break;
                }
            }

            const int edgeSpan = distNeg + distPos;
            float edgeBlend = 0.0f;
            if (edgeSpan > 0) {
                edgeBlend = std::abs(static_cast<float>(distPos - distNeg)) / static_cast<float>(edgeSpan);
            }

            edgeBlend = std::clamp(edgeBlend, 0.0f, 1.0f);

            // --------------------------------------------------
            // Sub-pixel AA
            // --------------------------------------------------
            const float lumaAvg =
                (lumaN + lumaS + lumaW + lumaE) * 0.25f;

            const float lumaDeltaML =
                std::abs(lumaAvg - lumaM);

            float subpixBlend = lumaContrast > 1e-5f
                ? lumaDeltaML / lumaContrast
                : 0.0f;

            subpixBlend = std::clamp(subpixBlend, 0.0f, 1.0f);
            subpixBlend = subpixBlend * subpixBlend;
            subpixBlend = std::min(subpixBlend, FXAA_QUALITY_SUBPIX_CAP);

            // --------------------------------------------------
            // Final blend
            // --------------------------------------------------
            const float finalBlend = std::max(edgeBlend, subpixBlend);

            const size_t offsetA = isHorizontal ? -w : -1;
            const size_t offsetB = isHorizontal ?  w :  1;

            const auto& [cMr, cMg, cMb, _1] = inBuffer[cordM];
            const auto& [cAr, cAg, cAb, _2] = inBuffer[cordM + offsetA];
            const auto& [cBr, cBg, cBb, _3] = inBuffer[cordM + offsetB];

            const float fr = (cAr + cBr) * 0.5f;
            const float fg = (cAg + cBg) * 0.5f;
            const float fb = (cAb + cBb) * 0.5f;

            FloatPixel out{};
            out.r = cMr * (1.0f - finalBlend) + fr * finalBlend;
            out.g = cMg * (1.0f - finalBlend) + fg * finalBlend;
            out.b = cMb * (1.0f - finalBlend) + fb * finalBlend;

            outBuffer[cordM] = out;
        }
    }
}

void Graphic::FXAAC(std::vector<FloatPixel>& inBuffer,
                           std::vector<FloatPixel>& outBuffer) const {
    const int w = static_cast<int>(engine->width);
    const int h = static_cast<int>(engine->height);

    // 预计算亮度
    for (auto& [r, g, b, i] : inBuffer) {
        i = 0.213f * r + 0.715f * g + 0.072f * b;
    }

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {

            const size_t cordM = x + y * w;
            const size_t cordN = cordM - w;
            const size_t cordS = cordM + w;
            const size_t cordW = cordM - 1;
            const size_t cordE = cordM + 1;

            const float lumaM = inBuffer[cordM].i;
            const float lumaN = inBuffer[cordN].i;
            const float lumaS = inBuffer[cordS].i;
            const float lumaW = inBuffer[cordW].i;
            const float lumaE = inBuffer[cordE].i;

            const float lumaMin = std::min({lumaM, lumaN, lumaS, lumaW, lumaE});
            const float lumaMax = std::max({lumaM, lumaN, lumaS, lumaW, lumaE});
            const float lumaContrast = lumaMax - lumaMin;

            const float edgeThreshold = std::max(
                FXAA_CONSOLE_EDGE_THRESHOLD_MIN,
                lumaMax * FXAA_CONSOLE_EDGE_THRESHOLD
            );

            if (lumaContrast < edgeThreshold) {
                outBuffer[cordM] = inBuffer[cordM];
                continue;
            }

            // --------------------------------------------------
            // Edge orientation
            // --------------------------------------------------
            const float gradH = std::abs(lumaW - lumaE);
            const float gradV = std::abs(lumaN - lumaS);
            const bool isHorizontal = gradH >= gradV;

            const int step = isHorizontal ? w : 1;

            // --------------------------------------------------
            // Fixed edge search (2 steps, deterministic)
            // --------------------------------------------------
            int distNeg = 0;
            int distPos = 0;

            {
                if (const size_t c = cordM - step; std::abs(inBuffer[c].i - lumaM) >= lumaContrast * 0.5f)
                    distNeg = 1;
            }

            {
                if (const size_t c = cordM + step; std::abs(inBuffer[c].i - lumaM) >= lumaContrast * 0.5f)
                    distPos = 1;
            }

            float edgeBlend = 0.0f;
            if (distNeg || distPos) {
                edgeBlend = std::abs(static_cast<float>(distPos - distNeg));
            }

            edgeBlend = std::clamp(edgeBlend, 0.0f, 1.0f);
            const float lumaAvg =
                (lumaN + lumaS + lumaW + lumaE) * 0.25f;

            const float lumaDeltaML =
                std::abs(lumaAvg - lumaM);

            float subpixBlend = lumaContrast > 1e-5f
                ? lumaDeltaML / lumaContrast
                : 0.0f;

            subpixBlend = std::clamp(subpixBlend, 0.0f, 1.0f);
            subpixBlend = subpixBlend * subpixBlend;
            subpixBlend = std::min(subpixBlend, FXAA_CONSOLE_SUBPIX_CAP);

            // --------------------------------------------------
            // Final blend (Console稳定优先)
            // --------------------------------------------------
            const float finalBlend = std::max(edgeBlend, subpixBlend);

            const size_t offsetA = isHorizontal ? -w : -1;
            const size_t offsetB = isHorizontal ?  w :  1;

            const auto& [cMr, cMg, cMb, _1] = inBuffer[cordM];
            const auto& [cAr, cAg, cAb, _2] = inBuffer[cordM + offsetA];
            const auto& [cBr, cBg, cBb, _3] = inBuffer[cordM + offsetB];

            const float fr = (cAr + cBr) * 0.5f;
            const float fg = (cAg + cBg) * 0.5f;
            const float fb = (cAb + cBb) * 0.5f;

            FloatPixel out{};
            out.r = cMr * (1.0f - finalBlend) + fr * finalBlend;
            out.g = cMg * (1.0f - finalBlend) + fg * finalBlend;
            out.b = cMb * (1.0f - finalBlend) + fb * finalBlend;

            outBuffer[cordM] = out;
        }
    }
}
