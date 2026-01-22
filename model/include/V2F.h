//
// Created by 冬榆 on 2025/12/30.
//

#ifndef UNTITLED_VSOUT_H
#define UNTITLED_VSOUT_H

#include "Film.h"
#include "Vec.hpp"
#include "VecPro.hpp"


struct V2F{
    Vec4 worldPosi;  // 世界空间坐标
    Vec4 clipPosi;   // 裁剪空间坐标
    Vec3 normal;     // 法向量
    VecN<2> uv;         // 纹理坐标
    Vec3 VexLightF;  // 顶点颜色(float, 用于VexLight)
    std::array<Vec3,3> PixLightOri;  // 逐像素光源朝向
    Vec3 MainLightOri;  // 主光源朝向
    Vec3 CameraOri;     // 相机朝向

    float invW{};  // 裁剪空间中的1/w
    bool alive{};  // 标记是否被裁剪

    V2F(const Vec4 &world, const Vec4 &clip, const Vec3 &normal, const VecN<2> &uv, const float &invW);
    V2F() = default;
};


#endif //UNTITLED_VSOUT_H