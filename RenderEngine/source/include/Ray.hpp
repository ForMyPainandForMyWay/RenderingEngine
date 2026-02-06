//
// Created by 冬榆 on 2026/1/23.
//

#ifndef RENDERINGENGINE_RAY_HPP
#define RENDERINGENGINE_RAY_HPP
#include "MatPro.hpp"

struct Ray {
    Vec4 orignPosi;
    Vec4 Direction;
};

// transformRay 不对 dir 进行 normalize
// 模型空间求出的 t 直接等于世界空间的 t 值。
Ray TransformRayToModel(const Ray& worldRay, const Mat4& invMat);

#endif //RENDERINGENGINE_RAY_HPP