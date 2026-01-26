//
// Created by 冬榆 on 2025/12/26.
//

#include "Lights.h"

Lights::Lights() {
    this->LightType = Ambient;
    this->tf.setPosition(0.0f, 0.0f, 0.0f);
}

Lights::Lights(const LType type, const float x, const float y, const float z) {
    this->LightType = type;
    this->tf.setPosition(x, y, z);
}

int Lights::getLType() const {
    return LightType;
}

void Lights::setColor(const uint8_t r, const uint8_t g,
                      const uint8_t b, const uint8_t a) {
    this->color = Pixel(r, g, b, a);
}

void Lights::setI(const float i) {
    this->intensity = i;
}

void Lights::updateP(const Vec3 &translate) {
    this->tf.multP(translate);
}

void Lights::updateQ(const Vec4 &quaternion) {
    this->tf.multQ(quaternion);
}

void Lights::updateS(const Vec3 &scale) {
    this->tf.multS(scale);
}

const Mat4& MainLight::ViewMat() {
    return this->tf.getViewMat();
}

const Mat4& MainLight::ProjectionMat() {
    if (ProjIsDirty) {
        this->updateProject();
        this->ProjIsDirty = false;
    }
    return this->Projection;
}

// 更新投影矩阵，不进行标记位更新
void MainLight::updateProject() {
    if (LightType == Spot) {
        // 透视投影（Spot 光）
        Mat4 P(0.0f);
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
    }
    else if (LightType == Direct) {
        // 正交投影（Directional 光）
        Mat4 P(0.0f);
        const float l = Left;
        const float r = Right;
        const float b = Bottom;
        const float t = Top;
        const float n = NearPlane;
        const float f = FarPlane;
        // 正交矩阵公式
        P[0][0] = 2.0f / (r - l);
        P[1][1] = 2.0f / (t - b);
        P[2][2] = -2.0f / (f - n);
        P[3][3] = 1.0f;
        P[0][3] = -(r + l) / (r - l);
        P[1][3] = -(t + b) / (t - b);
        P[2][3] = -(f + n) / (f - n);
        Projection = P;
    }
}
