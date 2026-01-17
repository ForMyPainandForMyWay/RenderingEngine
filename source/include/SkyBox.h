//
// Created by 冬榆 on 2026/1/17.
//

#ifndef RENDERINGENGINE_SKYBOX_H
#define RENDERINGENGINE_SKYBOX_H

#include "RenderObjects.h"


class SkyBox : public RenderObjects{
public:
    SkyBox();
    [[nodiscard]] std::shared_ptr<Mesh> getMesh() const;
protected:
    std::shared_ptr<Mesh> mesh;
};


#endif //RENDERINGENGINE_SKYBOX_H