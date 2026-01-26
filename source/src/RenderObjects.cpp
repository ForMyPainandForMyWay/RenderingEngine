//
// Created by 冬榆 on 2025/12/26.
//

#include "RenderObjects.h"

#include "MathTool.hpp"
#include "Mesh.h"


RenderObjects::RenderObjects(const std::shared_ptr<Mesh>& m) {
    this->setMesh(m);
}
void RenderObjects::setMesh(const std::shared_ptr<Mesh> &m) {
    this->mesh = m;
}

// 仅返回Tf矩阵，自动更新计算
const Mat4& RenderObjects::ModelMat() {
    return this->tf.getWorldMat();
}

Mat4 RenderObjects::invModelMat() const{
    const Mat4 invR = Transpose(tf.getRMat());
    const Mat4 invS = diagMatInverse(tf.getSMat());
    const Mat4 invT = tf.getNegativeTMat();
    return invS * invR * invT;
}

// 不保证应用更新的方法
const Mat4& RenderObjects::ModelMatUnsafe() const{
    return tf.ModelMatrix;
}

const Mat4& RenderObjects::NormalMatUnsafe() const {
    return tf.NormalWorldMat;
}

// 用于计算法向量的Tf矩阵
const Mat4& RenderObjects::InverseTransposedMat() {
    return this->tf.getNormalWorldMat();
}

std::shared_ptr<Mesh> RenderObjects::getMesh() const {
    return this->mesh;
}

void RenderObjects::updateP(const Vec3 &translate) {
    this->tf.multP(translate);
}

void RenderObjects::updateQ(const Vec4 &quaternion) {
    this->tf.multQ(quaternion);
}

void RenderObjects::updateS(const Vec3 &scale) {
    this->tf.multS(scale);
}

// 更新计算MV并返回，需要传入PV矩阵
Mat4 RenderObjects::updateMVP(const Mat4 &PV) {
    if (this->isDirty) {
        this->isDirty = false;
        this->MVP = PV * ModelMat();
    }
    return MVP;
}
