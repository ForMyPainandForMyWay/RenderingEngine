//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_CAMERA_H
#define UNTITLED_CAMERA_H

#include "Vec.hpp"
#include "Mat.hpp"
#include "Transform.h"


class Camera{
public:
    Camera();
    void setParameters(float F, float Near, float Far, float A);

    const MatMN<4, 4>& ViewMat();        // 视角变换矩阵
    const MatMN<4, 4>& ProjectionMat();  // 投影矩阵

    void updateProject();  // 更新投影变换矩阵
    void updateP(const VecN<3> &translate);   // 更新位姿
    void updateQ(const VecN<4> &quaternion);  // 更新旋转
    void updateS(const VecN<3> &scale);       // 放缩

private:
    float FOV = 45;        // 视场角,单位为度
    float NearPlane = 0.1f;  // 近平面
    float FarPlane = 100.0f;  // 远平面
    float AspectRatio = 1.0;  // 屏幕 宽/高,画面比例
    bool ProjIsDirty = true;      // 修改相机参数脏位
    CameraTransform tf;      // 视角变换
    MatMN<4, 4> Projection;  // 投影变换矩阵
    VecN<3> up;              // 摄像头上方向
};


#endif //UNTITLED_CAMERA_H