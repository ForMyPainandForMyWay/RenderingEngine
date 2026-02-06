//
// Created by 冬榆 on 2026/1/9.
//

#ifndef RENDERINGENGINE_SHADOWMAP_H
#define RENDERINGENGINE_SHADOWMAP_H
#include <vector>


struct ShadowMap {
    ShadowMap(size_t width, size_t height);
    std::vector<float> ZBufferShadow;
    float& operator[](size_t i);
    const float& operator[](size_t i) const;
    void clear();
    void resize(size_t w, size_t h);
    void save() const;

    [[nodiscard]] float Sample(float u, float v) const;
    [[nodiscard]] float SamplePCF(float currentDepth, float bias, float u, float v, int R=1) const;
    [[nodiscard]] float SamplePCSS(
        float currentDepth,
        float bias,
        float u,
        float v,
        float lightSizeUV,     // 光源半径（UV空间）
        int   blockerSearchR,  // blocker 搜索半径（像素）
        int   minPCFR,         // 最小 PCF 半径
        int   maxPCFR          // 最大 PCF 半径
        ) const;
    bool FindBlocker(
        float currentDepth,
        float bias,
        int cx,
        int cy,
        int searchR,
        float& avgBlockerDepth,
        float& blockerCount
        ) const;
    size_t width, height;
};


#endif //RENDERINGENGINE_SHADOWMAP_H