//
// Created by 冬榆 on 2026/1/3.
//

#ifndef RENDERINGENGINE_F2P_H
#define RENDERINGENGINE_F2P_H

#include "Film.h"
#include "Vec.hpp"


struct F2P {
    size_t x{}, y{};
    Pixel Albedo = {255, 255, 255, 255};
    VecN<4> worldPosi;
    VecN<3> normal;
    float depth = 0;
    bool alive = true;
    void drop() { alive = false; }
    void keep() { alive = true; }
};


#endif //RENDERINGENGINE_F2P_H