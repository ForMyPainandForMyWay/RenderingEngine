//
// Created by 冬榆 on 2026/1/9.
//

#ifndef RENDERINGENGINE_GBUFFER_H
#define RENDERINGENGINE_GBUFFER_H

#include <unordered_map>

#include "Film.hpp"


struct Material;
struct Fragment;

struct GBufferData {
    Vec3 normal;
    Vec4 worldPosi;
};

struct GBuffer {
    GBuffer(size_t w, size_t h);
    size_t w{}, h{};
    std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment>> FragMap;
    std::vector<GBufferData> Gdata; // 存储每个像素的法线和世界坐标数据
    void clear();
};


#endif //RENDERINGENGINE_GBUFFER_H