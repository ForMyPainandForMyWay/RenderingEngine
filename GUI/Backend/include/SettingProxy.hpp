//
// Created by 冬榆 on 2026/2/15.
//

#ifndef MYQTAPP_SETTINGPROXY_HPP
#define MYQTAPP_SETTINGPROXY_HPP
#include <qobject.h>

class IEngine;

class SettingProxy : public QObject{
    Q_OBJECT
    Q_PROPERTY(int TriangleNums READ TriangleNums NOTIFY TriangleNumsChanged)
    Q_PROPERTY(int VertexNums READ VertexNums NOTIFY VertexNumsChanged)

public:
    explicit SettingProxy(IEngine* engine, QObject *parent = nullptr);
    ~SettingProxy() override;
    IEngine* engine{};

public slots:
    // 全局渲染设置
    void setRenderMode(const bool &isRaster);
    void setFXAA(const int &FXAALevel);
    void enableSkyBox(const bool &enableSky);
    void enableSSAO(const bool &enableSSAO);
    void enableShadow(const bool &enableShadow);
    void openObj(const QUrl &url);
    static  void saveImg(const QUrl &url);

    // 统计数据
    [[nodiscard]] int TriangleNums() const;
    [[nodiscard]] int VertexNums() const;

    // 灯光设置
    void enableEnv(const bool &enableEnv);
    void enableSpot(const bool &enableSpot);
    void enablePoint(const bool &enablePoint);

    static void setPitch(const float &pitch);
    static void setYaw(const float &yaw);

    // 相机设置
    static void setFOV(const float &FOV);
    static void setNear(const float &near);
    static void setFar(const float &far);

signals:
    void TriangleNumsChanged();
    void VertexNumsChanged();

private:
    int triNums = 0;
    int vexNums = 0;

    bool currentMode = true;
    int currentFXAA = 0;
    bool currentSky = false;
    bool currentSSAO = false;
    bool currentShadow = false;

    bool currentEnv = false;
    bool currentSpot = false;
    bool currentPoint = false;
};


#endif //MYQTAPP_SETTINGPROXY_HPP