//
// Created by 冬榆 on 2025/12/31.
//

#include "Uniform.h"

Uniform::Uniform(const MatMN<4, 4> &mvpM, const MatMN<4, 4> &normalM) {
    this->MVP = mvpM;
    this->normalTfMat = normalM;
}
