//
// Created by 冬榆 on 2025/12/26.
//

#include "Transform.h"
#include "MathTool.hpp"


// 注意：初始构造变换类时并不会计算变换矩阵
Transform::Transform() {
     this->position = Vec3();
     this->quaternion = Vec4();
     this->quaternion[3] = 1;  // 表示无旋转的四元数
     this->scale = Vec3(1.0f);
     this->isDirty = true;
}

// 设置四元数,内部检查模长并自动归一化
void Transform::setQuaternion(const float x, const float y,
                                const float z, const float w) {
     this->quaternion[0] = x, this->quaternion[1] = y;
     this->quaternion[2] = z, this->quaternion[3] = w;
     if ( abs((x*x + y*y + z*z + w*w) - 1) > 1e-6 )
          this->quaternion = normalize(quaternion);
     isDirty = true;
}

void Transform::setPosition(const float x, const float y, const float z) {
     this->position[0] = x, this->position[1] = y, this->position[2] = z;
     isDirty = true;
}

void Transform::setScale(const float x, const float y, const float z) {
     this->scale[0] = x, this->scale[1] = y, this->scale[2] = z;
     isDirty = true;
}

void Transform::setQ(const Vec4 newQ) {
     this->quaternion = newQ;
     isDirty = true;
}

void Transform::setP(const Vec3 newT) {
     this->position = newT;
     isDirty = true;
}

void Transform::setS(const Vec3 newS) {
     this->scale = newS;
     isDirty = true;
}

void Transform::multQ(const Vec4& deltaQ) {
     // 当前四元数
     const float& x1 = quaternion[0];
     const float& y1 = quaternion[1];
     const float& z1 = quaternion[2];
     const float& w1 = quaternion[3];

     // 增量四元数
     const float& x2 = deltaQ[0];
     const float& y2 = deltaQ[1];
     const float& z2 = deltaQ[2];
     const float& w2 = deltaQ[3];

     // 四元数乘法（Hamilton 乘法）
     const float x = w1*x2 + x1*w2 + y1*z2 - z1*y2;
     const float y = w1*y2 - x1*z2 + y1*w2 + z1*x2;
     const float z = w1*z2 + x1*y2 - y1*x2 + z1*w2;
     const float w = w1*w2 - x1*x2 - y1*y2 - z1*z2;
     this->setQuaternion(x,y,z,w);
     isDirty = true;
}

void Transform::multP(const Vec3 &deltaT) {
     this->position += deltaT;
     isDirty = true;
}

void Transform::multS(const Vec3 &deltaS) {
     scale = Hadamard(scale, deltaS);
     isDirty = true;
}

// 四元数计算旋转矩阵，虚部xyz在前，实部w在后
Mat4 Transform::getRMat() const{
     Mat4 R;
     for(size_t i = 0; i < 4; ++i) {
          R[i][i] = 1.0f;
     }
     const float x = quaternion[0];
     const float y = quaternion[1];
     const float z = quaternion[2];
     const float w = quaternion[3];

     const float xx = x*x, yy = y*y, zz = z*z;
     const float xy = x*y, xz = x*z, yz = y*z;
     const float wx = w*x, wy = w*y, wz = w*z;

     R[0][0] = 1 - 2*(yy + zz);
     R[0][1] = 2*(xy - wz);
     R[0][2] = 2*(xz + wy);

     R[1][0] = 2*(xy + wz);
     R[1][1] = 1 - 2*(xx + zz);
     R[1][2] = 2*(yz - wx);

     R[2][0] = 2*(xz - wy);
     R[2][1] = 2*(yz + wx);
     R[2][2] = 1 - 2*(xx + yy);
     return R;
}

Mat4 Transform::getTMat() const{
     Mat4 T;  // 平移矩阵
     // 初始化单位矩阵
     for(size_t i = 0; i < 4; ++i) T[i][i] = 1.0f;

     // 设置平移向量
     for(size_t i = 0; i < 3; ++i)
          T[i][3] = position[i];  // 行优先存储
     return T;
}

Mat4 Transform::getSMat() const{
     Mat4 S;
     for (size_t i = 0; i < 3; i++) S[i][i] = scale[i];
     S[3][3] = 1;
     return S;
}


ObjTransform::ObjTransform() {
     this->ModelMatrix = Mat4(0.0f);
     for (size_t i = 0; i < 4; ++i) {
          ModelMatrix[i][i] = 1;
     }
}

void ObjTransform::update() {
     if (isDirty) {
          const auto T = getTMat();
          const auto R = getRMat();
          const auto S = getSMat();
          this->ModelMatrix = T*R*S;
          this->NormalWorldMat = R * diagMatInverse(S);
     }
}

// 世界坐标变换矩阵，自动更新
const Mat4& ObjTransform::getWorldMat() {
     if (isDirty) {
          this->update();
          this->isDirty = false;
     }
     return this->ModelMatrix;
}

// 法向量使用的世界坐标，自动更新
const Mat4 &ObjTransform::getNormalWorldMat() {
     if (isDirty) {
          this->update();
          this->isDirty = false;
     }
     return this->NormalWorldMat;
}


// 用于计算-P得出的位移矩阵
Mat4 Transform::getNegativeTMat() const{
     Mat4 T;  // 平移矩阵
     // 初始化单位矩阵
     for(size_t i = 0; i < 4; ++i) T[i][i] = 1.0f;
     // 设置平移向量
     for(size_t i = 0; i < 3; ++i)
          T[i][3] = -this->position[i];  // 行优先存储
     return T;
}

void CameraTransform::update() {
     // 视角变换
     if (isDirty)
          this->ViewMatrix = Transpose(getRMat()) * getNegativeTMat();
}

const Mat4& CameraTransform::getViewMat() {
     if (isDirty) {
          this->update();
          this->isDirty = false;
     }
     return this->ViewMatrix;
}

const Vec3& CameraTransform::getPosition() const {
     return position;
}