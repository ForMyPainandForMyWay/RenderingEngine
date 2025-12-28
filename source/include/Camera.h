//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_CAMERA_H
#define UNTITLED_CAMERA_H
#include "Transform.h"
#include "Vec.hpp"


class Camera{
public:
    Camera();
    void setParameters(float F, float Near, float Far, float A);

    const MatMN<4, 4>& ViewMat();  // 视角变换矩阵
    const MatMN<4, 4>& ProjectionMat();  // 投影矩阵

    // 更新投影变换矩阵
    void updateProject();

private:
    float FOV = 45;  // 视场角,单位为度
    float NearPlane = 20;   // 近平面
    float FarPlane = 100;  // 远平面
    float AspectRatio = 4.0/3.0;  // 屏幕 宽/高,画面比例

    bool ProjIsDirty = true;  // 修改相机参数脏位
    CameraTransform tf;  // 视角变换
    MatMN<4, 4> Projection;  // 投影变换矩阵
    VecN<3> up;      // 摄像头上方向
};


#endif //UNTITLED_CAMERA_H