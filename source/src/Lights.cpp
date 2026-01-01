//
// Created by 冬榆 on 2025/12/26.
//

#include "Lights.h"

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

void Lights::updateP(const VecN<3> &translate) {
    this->tf.multP(translate);
}

void Lights::updateQ(const VecN<4> &quaternion) {
    this->tf.multQ(quaternion);
}

void Lights::updateS(const VecN<3> &scale) {
    this->tf.multS(scale);
}
