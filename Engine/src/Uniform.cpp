//
// Created by 冬榆 on 2025/12/31.
//

#include <iostream>

#include "Uniform.h"


Uniform::Uniform(const Mat4 &mM, const Mat4 &mvpM, const Mat4 &normalM) {
    this->M = mM;
    this->MVP = mvpM;
    this->normalTfMat = normalM;
}

GlobalUniform::GlobalUniform(const size_t width, const size_t height, const size_t wShadow, const size_t hShadow) {
    this->ViewPort = Mat4{};
    ViewPort[0][0] = static_cast<float>(width) / 2;
    ViewPort[0][3] = static_cast<float>(width) / 2;
    ViewPort[1][1] = -static_cast<float>(height) / 2;
    ViewPort[1][3] = static_cast<float>(height) / 2;
    ViewPort[2][2] = 1;
    ViewPort[3][3] = 1;
    setShadowViewPort(wShadow, hShadow);
    screenHeight = height;
    screenWidth = width;
}

const Mat4 &GlobalUniform::getScreenViewPort() const{
    return ViewPort;
}

const Mat4& GlobalUniform::getShadowViewPort() const {
    return ShadowViewPort;
}

const Mat4 &GlobalUniform::getShadowPV() const{
    return projectViewShadow;
}

const Mat4& GlobalUniform::getCameraView() const {
    return CmaeraView;
}

const Mat4& GlobalUniform::getCameraProj() const{
    return CmaeraProj;
}

void GlobalUniform::setCameraPos(const Vec3 &cameraPos) {
    this->CameraPos = cameraPos;
}

void GlobalUniform::setProjectViewShadow(const Mat4 &pv) {
    this->projectViewShadow = pv;
}

// 设置阴影视口变换矩阵
void GlobalUniform::setShadowViewPort(const size_t width, const size_t height) {
    ShadowViewPort = Mat4{};
    ShadowViewPort[0][0] = static_cast<float>(width) / 2;
    ShadowViewPort[0][3] = static_cast<float>(width) / 2;
    ShadowViewPort[1][1] = -static_cast<float>(height) / 2;
    ShadowViewPort[1][3] = static_cast<float>(height) / 2;
    ShadowViewPort[2][2] = 1;
    ShadowViewPort[3][3] = 1;
}

void GlobalUniform::setCameraViewM(const Mat4 &CameraView) {
    CmaeraView = CameraView;
}

void GlobalUniform::setCameraProjM(const Mat4 &CameraProj) {
    CmaeraProj = CameraProj;
}