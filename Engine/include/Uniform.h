//
// Created by 冬榆 on 2025/12/31.
//

#ifndef UNTITLED_UNIFORM_H
#define UNTITLED_UNIFORM_H

#include "Mat.hpp"


class Uniform {
public:
    Uniform(const MatMN<4, 4> &mM,
            const MatMN<4, 4> &mvpM,
            const MatMN<4, 4> &normalM);

    friend class BlinnShader;

protected:
    MatMN<4, 4> M;
    MatMN<4, 4> MVP;
    MatMN<4, 4> normalTfMat;
};


class GlobalUniform {
public:
    GlobalUniform(size_t width, size_t height, size_t wShadow, size_t hShadow);
    // GlobalUniform(MatMN<4, 4> &&pv, size_t width, size_t height);
    [[nodiscard]] const MatMN<4, 4>& getScreenViewPort() const;
    [[nodiscard]] const MatMN<4, 4>& getShadowViewPort() const;
    [[nodiscard]] const MatMN<4, 4>& getShadowPV() const;
    [[nodiscard]] const VecN<3>& getCameraPos() const { return CameraPos; }
    void setProjectView(const MatMN<4, 4> &pv);
    void setCameraPos(const VecN<3> &cameraPos);
    void setProjectViewShadow(const MatMN<4, 4> &pv);
    void setShadowViewPort(size_t width, size_t height);

protected:
    MatMN<4, 4> projectViewShadow;  // 光源的PV矩阵
    MatMN<4, 4> ShadowViewPort; // 光源视口变换矩阵
    MatMN<4, 4> ProjectView;  // 相机的PV矩阵
    MatMN<4, 4> ViewPort;     // 相机视口变换矩阵
    VecN<3> CameraPos;  // 相机的位置
};


#endif //UNTITLED_UNIFORM_H