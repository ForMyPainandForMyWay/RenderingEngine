//
// Created by 冬榆 on 2025/12/31.
//

#ifndef UNTITLED_UNIFORM_H
#define UNTITLED_UNIFORM_H

#include "Mat.hpp"


class Uniform {
public:
    Uniform(const MatMN<4, 4> &mvpM, const MatMN<4, 4> &normalM);

    friend class Shader;
protected:
    MatMN<4, 4> MVP;
    MatMN<4, 4> normalTfMat;
};


#endif //UNTITLED_UNIFORM_H