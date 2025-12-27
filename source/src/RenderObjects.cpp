//
// Created by 冬榆 on 2025/12/26.
//

#include "RenderObjects.h"


RenderObjects::RenderObjects(Mesh *m) {
    this->setMesh(m);
}

void RenderObjects::setMesh(Mesh *m) {
    this->mesh = m;
}
