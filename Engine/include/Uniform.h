//
// Created by 冬榆 on 2025/12/31.
//

#ifndef UNTITLED_UNIFORM_H
#define UNTITLED_UNIFORM_H

#include "Mat.hpp"


struct Uniform {
    Uniform(const MatMN<4, 4> &mM,
            const MatMN<4, 4> &mvpM,
            const MatMN<4, 4> &normalM);
    Uniform() = default;
    MatMN<4, 4> M;
    MatMN<4, 4> MVP;
    MatMN<4, 4> normalTfMat;
};


class GlobalUniform {
public:
    GlobalUniform(size_t width, size_t height, size_t wShadow, size_t hShadow);
    [[nodiscard]] const MatMN<4, 4>& getScreenViewPort() const;
    [[nodiscard]] const MatMN<4, 4>& getShadowViewPort() const;
    [[nodiscard]] const MatMN<4, 4>& getShadowPV() const;
    [[nodiscard]] const MatMN<4, 4>& getCameraView() const;
    [[nodiscard]] const MatMN<4, 4>& getCameraProj() const;
    [[nodiscard]] const VecN<3>& getCameraPos() const { return CameraPos; }
    void setCameraPos(const VecN<3> &cameraPos);
    void setProjectViewShadow(const MatMN<4, 4> &pv);
    void setShadowViewPort(size_t width, size_t height);
    void setCameraViewM(const MatMN<4, 4> &CameraView);
    void setCameraProjM(const MatMN<4, 4> &CameraView);
    size_t screenHeight{},screenWidth{};  // 屏幕大小

protected:
    MatMN<4, 4> projectViewShadow;  // 光源的PV矩阵
    MatMN<4, 4> ShadowViewPort; // 光源视口变换矩阵
    MatMN<4, 4> ViewPort;    // 相机视口变换矩阵
    MatMN<4, 4> CmaeraView;  // 相机视图矩阵
    MatMN<4, 4> CmaeraProj;  // 相机投影矩阵
    VecN<3> CameraPos;  // 相机的位置
};


#endif //UNTITLED_UNIFORM_H