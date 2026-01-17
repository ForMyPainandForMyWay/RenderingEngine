//
// Created by 冬榆 on 2026/1/9.
//

#ifndef RENDERINGENGINE_GBUFFER_H
#define RENDERINGENGINE_GBUFFER_H

#include "Film.h"


struct Material;
struct Fragment;


struct GBuffer {
    std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment>> FragMap;
    void clear();
};


#endif //RENDERINGENGINE_GBUFFER_H