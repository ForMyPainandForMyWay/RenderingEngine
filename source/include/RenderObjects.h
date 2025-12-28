//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_OBJECTS_H
#define UNTITLED_OBJECTS_H

#include "Mesh.h"
#include "Transform.h"


// 用于渲染的物体对象
class RenderObjects{
public:
    explicit RenderObjects(Mesh *m);
    void setMesh(Mesh *m);
    const MatMN<4,4>& ModelMat();

private:
    Mesh *mesh = nullptr;    // 所属网格模型
    ObjTransform tf;  // 模型变换M
    bool visible = true;     // 可见性
};


#endif //UNTITLED_OBJECTS_H