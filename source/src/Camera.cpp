//
// Created by 冬榆 on 2025/12/26.
//

#include "Camera.h"
#include "V2F.h"


Camera::Camera() {
    tf = CameraTransform();
    up = VecN<3>();
    up[1] = 1;
}

void Camera::setParameters(const float F, const float Near, const float Far, const float A) {
    FOV = F;
    NearPlane = Near;
    FarPlane = Far;
    AspectRatio = A;
    ProjIsDirty = true;
}

// 返回视角矩阵V，自动更新
const MatMN<4, 4>& Camera::ViewMat() {
    return tf.getViewMat();
}

// 返回投影矩阵P，自动更新
const MatMN<4, 4> &Camera::ProjectionMat() {
    if (ProjIsDirty) this->updateProject();
    return Projection;
}

MatMN<4, 4> Camera::invProjectionMat() const{
    MatMN<4,4> InvP(0.0f);

    const float rad = FOV * 0.5f * 3.1415926535f / 180.0f;
    const float f_val = 1.0f / std::tan(rad); // cot(FOV/2)
    const float n = NearPlane;
    const float fa = FarPlane;
    const float inv_f = 1.0f / f_val;
    const float denom = 2.0f * fa * n; // 2 * far * near
    InvP[0][0] = AspectRatio * inv_f;   // a / f
    InvP[1][1] = inv_f;                 // 1 / f
    InvP[2][3] = -1.0f;                 // 第3行第4列 = -1
    // 修正：这里必须是负号！
    InvP[3][2] = -(fa - n) / denom;     // -(far - near) / (2 * far * near)
    InvP[3][3] = (fa + n) / denom;      // (far + near) / (2 * far * near)
    return InvP;
}

MatMN<4, 4> Camera::RMat() const {
    return tf.getRMat();
}

void Camera::updateProject() {
    MatMN<4,4> P(0.0f);

    const float rad = FOV * 0.5f * 3.1415926535f / 180.0f;
    const float f = 1.0f / std::tan(rad);
    const float n = NearPlane;
    const float fa = FarPlane;

    P[0][0] = f / AspectRatio;
    P[1][1] = f;
    P[2][2] = -(fa + n) / (fa - n);
    P[2][3] = -(2.0f * fa * n) / (fa - n);
    P[3][2] = -1.0f;
    P[3][3] = 0.0f;

    Projection = P;
    ProjIsDirty = false;
}

void Camera::updateP(const VecN<3> &translate) {
    this->tf.multP(translate);
}

void Camera::updateQ(const VecN<4> &quaternion) {
    this->tf.multQ(quaternion);
}

void Camera::updateS(const VecN<3> &scale) {
    this->tf.multS(scale);
}