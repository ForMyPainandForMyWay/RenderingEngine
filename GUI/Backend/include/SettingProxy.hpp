//
// Created by 冬榆 on 2026/2/15.
//

#ifndef MYQTAPP_SETTINGPROXY_HPP
#define MYQTAPP_SETTINGPROXY_HPP
#include <qobject.h>

class Receiver;
class FrameProvider;
class IEngine;

class SettingProxy : public QObject{
    Q_OBJECT
    Q_PROPERTY(int TriangleNums READ TriangleNums NOTIFY TriangleNumsChanged)
    Q_PROPERTY(int VertexNums READ VertexNums NOTIFY VertexNumsChanged)

public:
    explicit SettingProxy(std::unique_ptr<IEngine> engine, FrameProvider* fp, QObject *parent = nullptr);
    ~SettingProxy() override;
    std::unique_ptr<IEngine> engine = nullptr;

public slots:
    // 全局渲染设置
    void setRenderMode(const bool &isRaster);
    void setFXAA(const int &FXAALevel);
    void enableSkyBox(const bool &enableSky);
    void enableSSAO(const bool &enableSSAO);
    void enableShadow(const bool &enableShadow);
    void openObj(const QUrl &url);
    void saveImg(const QUrl &url) const;

    // 统计数据
    [[nodiscard]] int TriangleNums() const;
    [[nodiscard]] int VertexNums() const;

    // 灯光设置
    void enableEnv(const bool &enableEnv);
    void enableSpot(const bool &enableSpot);
    void enablePoint(const bool &enablePoint);
    void setEnvColor(const float &r, const float &g, const float &b);
    void setSpotColor(const float &r, const float &g, const float &b);
    void setPointColor(const float &r, const float &g, const float &b);

    // 这里就先做成物体的旋转了
    void setPitch(const float &pitch);
    void setYaw(const float &yaw);

    // 相机设置
    void setFOV(const float &FOV) const;
    void setNear(const float &near) const;
    void setFar(const float &far) const;

signals:
    void TriangleNumsChanged();
    void VertexNumsChanged();

private:
    int objID = -1;

    FrameProvider* fp = nullptr;  // 引用不拥有
    Receiver* currentRecv = nullptr;  // 拥有
    int triNums = 0;
    int vexNums = 0;

    bool currentMode = true;  // 渲染模式，true为光栅模式，false为光追模式
    uint8_t currentFXAA = 0;  // FXAA档位,0-3为从无到高
    bool currentSky = false;
    bool currentSSAO = false;
    bool currentShadow = false;

    bool currentEnv = true;
    bool currentSpot = false;
    bool currentPoint = false;

    uint8_t envCorlor[3] = {100, 100, 100};
    uint8_t spotColor[3] = {255, 255, 255};
    uint8_t pointColor[3] = {255, 255, 255};

    float currentYaw = 0.5f;
    float currentPitch = 0.5f;
};


#endif //MYQTAPP_SETTINGPROXY_HPP