//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_LIGHTS_H
#define UNTITLED_LIGHTS_H

#include "Film.h"
#include "MatPro.hpp"
#include "Transform.h"


struct TextureMap;
enum LType{Direct, Point, Spot, Ambient};


// 不同类型的光源
class Lights {
    // 光源类型:平行光、点光源、聚光灯、环境光
    // 注意平行光与环境光无需变换,聚光灯需要额外指定朝向
    // 由于点光源做阴影开销过大，因此只支持平行光和聚光灯阴影
public:
    Lights();
    Lights(LType type, float x, float y, float z);
    [[nodiscard]] int getLType() const;
    void setColor(uint8_t r = 255, uint8_t g = 255,
                  uint8_t b = 255, uint8_t a = 255);
    void setI(float i = 1);

    void updateP(const Vec3 &translate);   // 更新位姿
    void updateQ(const Vec4 &quaternion);  // 更新旋转
    void updateS(const Vec3 &scale);  // 放缩

    [[nodiscard]] float getI() const { return intensity; }
    [[nodiscard]] Vec3 getPosi() const { return tf.getPosition(); }
    [[nodiscard]] Pixel getColor() const { return color; }

    bool alive = false;

protected:
    LType LightType = Ambient;
    LightTransform tf;
    Pixel color = {255, 255, 255, 255};
    float intensity = 2;   // 光照强度
    float range = 0;       // 点光/聚光衰减
    // 聚光灯
    float FOV = 45;  // 聚光灯光锥角度
    float NearPlane = 0.1f;
    float FarPlane = 20.0f;
    float AspectRatio = 1.0f;  // 聚光灯纵横比
    // 方向光
    float Left = -10.0f;
    float Right = 10.0f;
    float Top = 10.0f;
    float Bottom = -10.0f;
};

class MainLight: public Lights {
public:
    const Mat4& ViewMat();        // 视角变换矩阵
    const Mat4& ProjectionMat();  // 返回投影矩阵P，自动更新
    void updateProject();  // 更新投影变换矩阵

    friend class Engine;

protected:
    enum LType {Direct, Spot};
    LType LightType = Spot;
    Mat4 Projection;  // 投影变换矩阵
    bool ProjIsDirty = true;  // 修改光源参数脏位
};

class EnvironmentLight: public Lights {
public:
    EnvironmentLight() { LightType = Ambient; }

protected:
    float Intensity = 1.0f;  // 环境光强度
    TextureMap *EnvMap = nullptr;  // IBL环境贴图
};


#endif //UNTITLED_LIGHTS_H