//
// Created by 冬榆 on 2026/1/23.
//

#include "Ray.hpp"

Ray TransformRayToModel(const Ray& worldRay, const Mat4& invMat) {
    Ray localRay;
    localRay.orignPosi = invMat * worldRay.orignPosi;
    localRay.Direction = invMat * worldRay.Direction;
    return localRay;
}