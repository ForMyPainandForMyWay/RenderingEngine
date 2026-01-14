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
    size_t width, height;
};


#endif //RENDERINGENGINE_SHADOWMAP_H