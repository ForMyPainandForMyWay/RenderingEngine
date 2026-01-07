//
// Created by 冬榆 on 2025/12/26.
//

#include "RenderObjects.h"


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

// 用于计算法向量的Tf矩阵
const MatMN<4, 4>& RenderObjects::InverseTransposedMat() {
    return this->tf.getNormalWorldMat();
}

const Mesh* RenderObjects::getMesh() const {
    return this->mesh;
}

void RenderObjects::updateP(const VecN<3> &translate) {
    this->tf.multP(translate);
}

void RenderObjects::updateQ(const VecN<4> &quaternion) {
    this->tf.multQ(quaternion);
}

void RenderObjects::updateS(const VecN<3> &scale) {
    this->tf.multS(scale);
}

// 更新计算MV并返回，需要传入PV矩阵
MatMN<4, 4> RenderObjects::updateMVP(const MatMN<4, 4> &PV) {
    if (this->isDirty) {
        this->isDirty = false;
        this->MVP = PV * ModelMat();
    }
    return MVP;
}
