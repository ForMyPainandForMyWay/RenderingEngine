//
// Created by 冬榆 on 2026/1/9.
//

#ifndef RENDERINGENGINE_GBUFFER_H
#define RENDERINGENGINE_GBUFFER_H

#include "Film.h"


struct Material;
struct Fragment;


struct GBuffer {
    std::unordered_map<Material*, std::vector<Fragment>> FragMap;
};


#endif //RENDERINGENGINE_GBUFFER_H