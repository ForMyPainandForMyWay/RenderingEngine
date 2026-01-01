//
// Created by 冬榆 on 2025/12/30.
//

#ifndef UNTITLED_VSOUT_H
#define UNTITLED_VSOUT_H

#include "Film.h"
#include "Vec.hpp"


struct V2F{
    VecN<4> position;  // 物理坐标
    VecN<3> normal;    // 法向量
    VecN<2> uv;        // 纹理坐标
    Pixel pix{};      // 顶点颜色

    float invW{};  // 裁剪空间中的1/w
    bool alive{};         // 标记是否被裁剪
    V2F(const VecN<4> &clip, const VecN<4> &normal, const VecN<2> &uv, const float &invW);
    V2F() = default;

    bool operator==(const V2F &other) const {
        return position == other.position &&
                 normal == other.normal &&
                     uv == other.uv;
    }
};


#endif //UNTITLED_VSOUT_H