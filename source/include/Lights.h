//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_LIGHTS_H
#define UNTITLED_LIGHTS_H
#include "film.h"
#include "Transform.h"


// 不同类型的光源
class Lights {
    // 光源类型:平行光、点光源、聚光灯、环境光
    enum LightType{Direct, Point, Spot, Ambient};
    LightType LightType = Ambient;
    ObjTransform Ltf;
    Pixel color = {255, 255, 255, 255};
    float intensity = 1;   // 光照强度
    float range = 0;       // 点光/聚光衰减
    float spotAngle = 0;  // 聚光灯角度
};


#endif //UNTITLED_LIGHTS_H