//
// Created by 冬榆 on 2025/12/31.
//

#ifndef UNTITLED_UNIFORM_H
#define UNTITLED_UNIFORM_H

#include "Mat.hpp"
#include "MatPro.hpp"


struct Uniform {
    Uniform(const Mat4 &mM,
            const Mat4 &mvpM,
            const Mat4 &normalM);
    Uniform() = default;
    Mat4 M;
    Mat4 MVP;
    Mat4 normalTfMat;
};


class GlobalUniform {
public:
    GlobalUniform(size_t width, size_t height, size_t wShadow, size_t hShadow);
    [[nodiscard]] const Mat4& getScreenViewPort() const;
    [[nodiscard]] const Mat4& getShadowViewPort() const;
    [[nodiscard]] const Mat4& getShadowPV() const;
    [[nodiscard]] const Mat4& getCameraView() const;
    [[nodiscard]] const Mat4& getCameraProj() const;
    [[nodiscard]] const Vec3& getCameraPos() const { return CameraPos; }
    void setCameraPos(const Vec3 &cameraPos);
    void setProjectViewShadow(const Mat4 &pv);
    void setShadowViewPort(size_t width, size_t height);
    void setCameraViewM(const Mat4 &CameraView);
    void setCameraProjM(const Mat4 &CameraProj);
    size_t screenHeight{},screenWidth{};  // 屏幕大小

protected:
    Mat4 projectViewShadow;  // 光源的PV矩阵
    Mat4 ShadowViewPort; // 光源视口变换矩阵
    Mat4 ViewPort;    // 相机视口变换矩阵
    Mat4 CmaeraView;  // 相机视图矩阵
    Mat4 CmaeraProj;  // 相机投影矩阵
    Vec3 CameraPos;  // 相机的位置
};


#endif //UNTITLED_UNIFORM_H