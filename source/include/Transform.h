//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_TRANSFORM_H
#define UNTITLED_TRANSFORM_H
#include "Mat.hpp"
#include "Vec.hpp"


class Transform {
public:
    Transform();

    void setQuaternion(float x, float y, float z, float w);
    void setPosition(float x, float y, float z);

    void updateTfMat();      // 更新变换矩阵
    [[nodiscard]] MatMN<4, 4> getTfMat() const;  // 获取变换矩阵(仅返回不计算)

    // 更新单一变换不更新tf
    void multR(const VecN<4> &deltaQ);  // 累积旋转,更新四元数
    void multT(const VecN<3> &deltaT);  // 累积位移,更新位移

protected:
    // 注：累计变换中,位移是加和,旋转与放缩是累乘
    VecN<3> position;
    VecN<4> quaternion;  // 四元数,注意四元素要满足归一化条件
    MatMN<4, 4> tf;      // 变换矩阵
};

// 带有scale的tf
class TransformWithScale : Transform{
public:
    TransformWithScale();
    void setScale(float x, float y, float z);
    void updateTfMat();      // 更新变换矩阵

    void multS(const VecN<3> &deltaS);  // 累积放缩,更新位移,注意非均匀放缩+旋转会对法线矩阵产生影响

protected:
    VecN<3> scale;
};

#endif //UNTITLED_TRANSFORM_H