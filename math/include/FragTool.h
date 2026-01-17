//
// Created by 冬榆 on 2026/1/7.
//

#ifndef RENDERINGENGINE_FRAGTOOL_H
#define RENDERINGENGINE_FRAGTOOL_H

#include "Vec.hpp"

struct TextureMap;
struct Pixel;

// 片元着色工具
Pixel Sample(const VecN<2>& uv, const std::shared_ptr<TextureMap>& texture);
Pixel BilinearSample(const VecN<2> &uv, const std::shared_ptr<TextureMap>& texture);
#endif //RENDERINGENGINE_FRAGTOOL_H