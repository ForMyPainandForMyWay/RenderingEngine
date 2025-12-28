//
// Created by 冬榆 on 2025/12/26.
//

#include "RenderObjects.h"

#include "Mat.hpp"


RenderObjects::RenderObjects(Mesh *m) {
    this->setMesh(m);
}

void RenderObjects::setMesh(Mesh *m) {
    this->mesh = m;
}

// 仅返回Tf矩阵，自动更新计算
const MatMN<4, 4>& RenderObjects::ModelMat() {
    return this->tf.getWorldMat();
}

