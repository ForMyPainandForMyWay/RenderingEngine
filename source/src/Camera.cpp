//
// Created by 冬榆 on 2025/12/26.
//

#include "Camera.h"

Camera::Camera() {
    tf = CameraTransform();
    up = VecN<3>();
    up[1] = 1;
}

void Camera::setParameters(float F, float Near, float Far, float A) {
    FOV = F;
    NearPlane = Near;
    FarPlane = Far;
    AspectRatio = A;
    ProjIsDirty = true;
}

const MatMN<4, 4>& Camera::ViewMat() {
    return tf.getViewMat();
}

const MatMN<4, 4> &Camera::ProjectionMat() {
    if (ProjIsDirty) this->updateProject();
    return Projection;
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

