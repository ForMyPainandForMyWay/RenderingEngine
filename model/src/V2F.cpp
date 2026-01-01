//
// Created by 冬榆 on 2025/12/30.
//

#include "V2F.h"

// 不进行齐次除法
V2F::V2F(const VecN<4> &clip, const VecN<4> &normal, const VecN<2> &uv, const float &invW) {
    this->position = clip;
    this->normal = VecN<3>{normal[0], normal[1], normal[2]};
    this->uv = uv;
    this->invW = invW;
}
