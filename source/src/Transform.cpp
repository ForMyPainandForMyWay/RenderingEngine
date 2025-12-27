//
// Created by 冬榆 on 2025/12/26.
//

#include "Transform.h"

#include <iostream>

// 注意：初始构造变换类时并不会计算变换矩阵
Transform::Transform() {
    this->position = VecN<3>();
    this->quaternion = VecN<4>();
    this->quaternion[0] = 1;
    // this->updateTfMat();
}

// 设置四元数,内部检查模长并自动归一化
void Transform::setQuaternion(const float x, const float y,
                              const float z, const float w) {
    if ( abs((x*x + y*y + z*z + w*w) - 1) > 1e6 )
        this->quaternion = normalize(quaternion);
    this->quaternion[0] = x, this->quaternion[1] = y;
    this->quaternion[2] = z, this->quaternion[3] = w;
}

void Transform::setPosition(const float x, const float y, const float z) {
    this->position[0] = x, this->position[1] = y, this->position[2] = z;
}

void Transform::updateTfMat() {
    // 注意这里依赖填充0的构造函数
    MatMN<4, 4> T;  // 平移矩阵
    MatMN<4, 4> R;  // 旋转矩阵

    // 注意矩阵按列优先存储
    for(size_t i = 0; i < 3; i++) {
        T[3][i] = this->position[i];
        R[i][i] = 1;  // R先按照单位矩阵初始化
    }
    T[3][3] = 1, R[3][3] = 1;

    const float x = quaternion[0];
    const float y = quaternion[1];
    const float z = quaternion[2];
    const float w = quaternion[3];

    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float wx = w * x;
    const float wy = w * y;
    const float wz = w * z;

    R[0][0] = 1.0f - 2.0f * (yy + zz);
    R[0][1] = 2.0f * (xy + wz);
    R[0][2] = 2.0f * (xz - wy);

    R[1][0] = 2.0f * (xy - wz);
    R[1][1] = 1.0f - 2.0f * (xx + zz);
    R[1][2] = 2.0f * (yz + wx);

    R[2][0] = 2.0f * (xz + wy);
    R[2][1] = 2.0f * (yz - wx);
    R[2][2] = 1.0f - 2.0f * (xx + yy);

    this->tf = T * R;
}

MatMN<4, 4> Transform::getTfMat() const {
    return this->tf;
}

void Transform::multR(const VecN<4>& deltaQ) {
    // 当前四元数
    const float& x1 = quaternion[0];
    const float& y1 = quaternion[1];
    const float& z1 = quaternion[2];
    const float& w1 = quaternion[3];

    // 增量四元数
    const float& x2 = deltaQ[0];
    const float& y2 = deltaQ[1];
    const float& z2 = deltaQ[2];
    const float& w2 = deltaQ[3];

    // 四元数乘法（Hamilton 乘法）: q_new = q_current * deltaQ
    const float x = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    const float y = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    const float z = w1*z2 + x1*y2 - y1*x2 + z1*w2;
    const float w = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    this->setQuaternion(x,y,z,w);
}


void Transform::multT(const VecN<3> &deltaT) {
    this->position += deltaT;
}


// 带Scale的变换,注意并不会计算变换矩阵
TransformWithScale::TransformWithScale(){
    this->scale = VecN<3>();
    setScale(1, 1, 1);
}

void TransformWithScale::setScale(const float x, const float y, const float z) {
    this->scale[0] = x, this->scale[1] = y, this->scale[2] = z;
}

void TransformWithScale::updateTfMat() {
    Transform::updateTfMat();
    MatMN<4, 4>S;
    for (size_t i = 0; i < 3; i++) S[i][i] = scale[i];
    S[3][3] = 1;
    this->tf = this->tf * S;
}

void TransformWithScale::multS(const VecN<3> &deltaS) {
    this->scale *= deltaS;
}
