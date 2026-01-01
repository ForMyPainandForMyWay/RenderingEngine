//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_LIGHTS_H
#define UNTITLED_LIGHTS_H

#include "Film.h"
#include "Transform.h"


// 不同类型的光源
class Lights {
    // 光源类型:平行光、点光源、聚光灯、环境光
    // 注意平行光与环境光无需变换,聚光灯需要额外指定朝向
public:
    [[nodiscard]] int getLType() const;
    void setColor(uint8_t r = 255, uint8_t g = 255,
                  uint8_t b = 255, uint8_t a = 255);
    void setI(float i = 1);

    void updateP(const VecN<3> &translate);  // 更新位姿
    void updateQ(const VecN<4> &quaternion);  // 更新旋转
    void updateS(const VecN<3> &scale);  // 放缩

protected:
    enum LightType{Direct, Point, Spot, Ambient};
    LightType LightType = Ambient;
    ObjTransform tf;
    Pixel color = {255, 255, 255, 255};
    float intensity = 1;   // 光照强度
    float range = 0;       // 点光/聚光衰减
    float spotAngle = 0;  // 聚光灯角度
};


#endif //UNTITLED_LIGHTS_H