//
// Created by 冬榆 on 2026/1/24.
//

#include "HitInfo.hpp"
#include "Ray.hpp"

void HitInfo::trans2World(const Mat4& ModelMat, const Mat4& NormalWorldMat){
    hitPos = ModelMat * hitPos;
    hitPos = hitPos / hitPos[3];
    hitNormal = normalize(NormalWorldMat * hitNormal);
}