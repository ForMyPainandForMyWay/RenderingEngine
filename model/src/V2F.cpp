//
// Created by 冬榆 on 2025/12/30.
//

#include "V2F.hpp"

// 不进行齐次除法
V2F::V2F(const Vec4 &world, const Vec4 &clip,
    const Vec3 &normal, const VecN<2> &uv,
    const float &invW) {
    this->worldPosi = world;
    this->clipPosi = clip;
    this->normal = Vec3{normal[0], normal[1], normal[2]};
    this->uv = uv;
    this->invW = invW;
}
