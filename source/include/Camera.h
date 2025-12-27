//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_CAMERA_H
#define UNTITLED_CAMERA_H
#include "Transform.h"
#include "Vec.hpp"


class Camera {
public:
    Camera();
    // void updateVtf

private:
    float FOV = 45;  // 视场角,单位为度
    float NearPlane = 20;   // 近平面
    float FarPlane = 100;  // 远平面
    float AspectRatio = 4.0/3.0;  // 屏幕 宽/高,画面比例

    Transform Vtf;   // 视角变换
    Transform Ptf;   // 投影变换
    VecN<3> up;      // 摄像头上方向
};


#endif //UNTITLED_CAMERA_H