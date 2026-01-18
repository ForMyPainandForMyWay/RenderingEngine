//
// Created by 冬榆 on 2026/1/3.
//

#ifndef RENDERINGENGINE_F2P_H
#define RENDERINGENGINE_F2P_H

#include "Film.h"


struct F2P {
    size_t x{}, y{};
    FloatPixel Albedo = {255, 255, 255};
    float depth = 0;
    bool alive = true;
    void drop() { alive = false; }
    void keep() { alive = true; }
};


#endif //RENDERINGENGINE_F2P_H