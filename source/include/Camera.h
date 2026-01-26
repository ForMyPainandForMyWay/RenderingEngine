//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_CAMERA_H
#define UNTITLED_CAMERA_H

#include "MatPro.hpp"
#include "Transform.h"


class Camera{
public:
    Camera();
    void setParameters(float F, float Near, float Far, float A);

    const Mat4& ViewMat();        // 视角变换矩阵
    const Mat4& ProjectionMat();  // 返回投影矩阵P，自动更新
    [[nodiscard]] Mat4 invProjectionMat() const;  // 返回投影矩阵P的逆矩阵
    [[nodiscard]] Mat4 RMat() const;            // 返回相机旋转矩阵
    [[nodiscard]] Vec3 getPosi() const { return tf.getPosition() ;}  // 返回相机位置
    [[nodiscard]] float getFov() const { return FOV; }
    [[nodiscard]] float getAspect() const { return AspectRatio; }
    [[nodiscard]] float getNear() const { return NearPlane; }
    [[nodiscard]] float getFar() const { return FarPlane; }

    void updateProject();  // 更新投影变换矩阵
    void updateP(const Vec3 &translate);   // 更新位姿
    void updateQ(const Vec4 &quaternion);  // 更新旋转
    void updateS(const Vec3 &scale);       // 放缩

private:
    float FOV = 45;        // 视场角,单位为度
    float NearPlane = 0.1f;  // 近平面
    float FarPlane = 20.0f;  // 远平面
    float AspectRatio = 1.0;  // 屏幕 宽/高,画面比例
    bool ProjIsDirty = true;      // 修改相机参数脏位
    CameraTransform tf;      // 视角变换
    Mat4 Projection;  // 投影变换矩阵
    Vec3 up;              // 摄像头上方向
};


#endif //UNTITLED_CAMERA_H