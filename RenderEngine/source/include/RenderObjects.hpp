//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_OBJECTS_H
#define UNTITLED_OBJECTS_H

#include <memory>

#include "Transform.hpp"

class Mesh;


// 用于渲染的物体对象
class RenderObjects{
public:
    RenderObjects() = default;
    explicit RenderObjects(const std::shared_ptr<Mesh>& m);
    void setMesh(const std::shared_ptr<Mesh> &m);
    const Mat4& ModelMat();
    [[nodiscard]] Mat4 invModelMat() const;  // 现场计算不存储
    [[nodiscard]] const Mat4& ModelMatUnsafe() const;
    [[nodiscard]] const Mat4& NormalMatUnsafe() const;
    const Mat4& InverseTransposedMat();
    [[nodiscard]] std::shared_ptr<Mesh> getMesh() const;

    // 位置变换接口
    void updateP(const Vec3 &translate);
    void updateQ(const Vec4 &quaternion);
    void updateS(const Vec3 &scale);

    // 更新MVP并返回
    Mat4 updateMVP(const Mat4 &PV);

protected:
    Mat4 MVP;
    ObjTransform tf;  // 模型变换M
    std::shared_ptr<Mesh> mesh;   // 网格模型指针
    bool visible = true;   // 可见性
    bool isDirty = true;   // MVP标记位
};


#endif //UNTITLED_OBJECTS_H