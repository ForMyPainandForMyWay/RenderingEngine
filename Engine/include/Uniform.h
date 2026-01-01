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


class GlobalUniform {
public:
    GlobalUniform(size_t width, size_t height);
    // GlobalUniform(MatMN<4, 4> &&pv, size_t width, size_t height);
    [[nodiscard]] const MatMN<4, 4>& getViewPort() const;
    void setProjectView(MatMN<4, 4> pv);

protected:
    MatMN<4, 4> ProjectView;  // PV矩阵
    MatMN<4, 4> ViewPort;     // 视口变换矩阵
};


#endif //UNTITLED_UNIFORM_H