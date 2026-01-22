//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_TRANSFORM_H
#define UNTITLED_TRANSFORM_H
#include "Mat.hpp"
#include "MatPro.hpp"
#include "Vec.hpp"



// 用于生成世界坐标、以及其他变换矩阵
class Transform{
public:
    virtual ~Transform() = default;
    Transform();

    // 直接设置状态
    void setQuaternion(float x, float y, float z, float w);
    void setPosition(float x, float y, float z);
    void setScale(float x, float y, float z);
    void setQ(Vec4 newQ);
    void setP(Vec3 newT);
    void setS(Vec3 newS);

    // 累积更新单一变换
    void multQ(const Vec4 &deltaQ);  // 累积旋转,更新四元数
    void multP(const Vec3 &deltaT);  // 累积位移,更新位移
    void multS(const Vec3 &deltaS);  // 累积放缩,更新位移,注意非均匀放缩+旋转会对法线矩阵产生影响

    [[nodiscard]] Mat4 getRMat() const;
    [[nodiscard]] Mat4 getTMat() const;
    [[nodiscard]] Mat4 getSMat() const;

    virtual void update()=0;  // 更新变换矩阵

protected:
    bool isDirty = true;  // 脏标记，用于记录变换矩阵是否需要更新
    Vec3 position;
    Vec4 quaternion;  // 四元数,注意四元素要满足归一化条件.以及内存布局w为实部在后
    Vec3 scale;
};


// 物体变换
class ObjTransform : public Transform{
public:
    ObjTransform();
    void update() override;
    const Mat4& getWorldMat();
    const Mat4& getNormalWorldMat();
protected:
    Mat4 ModelMatrix;
    Mat4 NormalWorldMat;
};


// 相机视角变换
class CameraTransform : public Transform{
public:
    Mat4 getNegativeTMat();  // 返回负位移向量构造的矩阵
    void update() override;
    const Mat4& getViewMat();  // 返回视角变换矩阵
    [[nodiscard]] const Vec3& getPosition() const;  // 返回相机位置
protected:
    Mat4 ViewMatrix;
};

using LightTransform = CameraTransform;

#endif //UNTITLED_TRANSFORM_H