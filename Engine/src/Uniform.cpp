//
// Created by 冬榆 on 2025/12/31.
//

#include "Uniform.h"

Uniform::Uniform(const MatMN<4, 4> &mvpM, const MatMN<4, 4> &normalM) {
    this->MVP = mvpM;
    this->normalTfMat = normalM;
}

GlobalUniform::GlobalUniform(const size_t width, const size_t height) {
    this->ViewPort = MatMN<4, 4>{};
    ViewPort[0][0] = static_cast<float>(width) / 2;
    ViewPort[0][3] = static_cast<float>(width) / 2;
    ViewPort[1][1] = static_cast<float>(height) / 2;
    ViewPort[1][3] = static_cast<float>(height) / 2;
    ViewPort[2][2] = 1;
    ViewPort[3][3] = 1;
}

const MatMN<4, 4> &GlobalUniform::getViewPort() const{
    return ViewPort;
}

void GlobalUniform::setProjectView(const MatMN<4, 4> pv) {
        this->ProjectView = pv;
}
