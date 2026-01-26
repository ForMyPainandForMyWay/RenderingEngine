//
// Created by 冬榆 on 2026/1/24.
//

#include "HitInfo.hpp"

void HitInfo::trans2World(const Mat4& ModelMat, const Mat4& NormalWorldMat){
    hitPos = ModelMat * hitPos;
    hitNormal = normalize(NormalWorldMat * hitNormal);
}