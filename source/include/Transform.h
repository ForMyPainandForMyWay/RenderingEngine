//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_TRANSFORM_H
#define UNTITLED_TRANSFORM_H
#include "Mat.hpp"
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
    void setQ(VecN<4> newQ);
    void setP(VecN<3> newT);
    void setS(VecN<3> newS);

    // 累积更新单一变换
    void multQ(const VecN<4> &deltaQ);  // 累积旋转,更新四元数
    void multP(const VecN<3> &deltaT);  // 累积位移,更新位移
    void multS(const VecN<3> &deltaS);  // 累积放缩,更新位移,注意非均匀放缩+旋转会对法线矩阵产生影响

    [[nodiscard]] MatMN<4, 4> getRMat() const;
    [[nodiscard]] MatMN<4, 4> getTMat() const;
    [[nodiscard]] MatMN<4, 4> getSMat() const;

    virtual void update()=0;  // 更新变换矩阵

protected:
    bool isDirty = true;  // 脏标记，用于记录变换矩阵是否需要更新
    VecN<3> position;
    VecN<4> quaternion;  // 四元数,注意四元素要满足归一化条件.以及内存布局w为实部在后
    VecN<3> scale;
};


// 物体变换
class ObjTransform : public Transform{
public:
    ObjTransform();
    void update() override;
    const MatMN<4, 4>& getWorldMat();
    const MatMN<4, 4>& getNormalWorldMat();
protected:
    MatMN<4, 4> ModelMatrix;
    MatMN<4, 4> NormalWorldMat;
};


// 相机视角变换
class CameraTransform : public Transform{
public:
    MatMN<4, 4> getNegativeTMat();  // 返回负位移向量构造的矩阵
    void update() override;
    const MatMN<4, 4>& getViewMat();  // 返回视角变换矩阵
    [[nodiscard]] const VecN<3>& getPosition() const;  // 返回相机位置
protected:
    MatMN<4, 4> ViewMatrix;
};

using LightTransform = CameraTransform;

#endif //UNTITLED_TRANSFORM_H