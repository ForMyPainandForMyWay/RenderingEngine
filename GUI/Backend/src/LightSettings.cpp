//
// Created by 冬榆 on 2026/2/15.
//

#include <QDebug>

#include "SettingProxy.hpp"

// 灯光设置
void SettingProxy::enableEnv(const bool &enableEnv) {
    if (enableEnv == currentEnv) return;
    currentEnv = enableEnv;
    qDebug() << "enableEnv: " << enableEnv;
    if (currentEnv) {

    } else {

    }
}

void SettingProxy::enableSpot(const bool &enableSpot) {
    if (enableSpot == currentSpot) return;
    currentSpot = enableSpot;
    qDebug() << "enableSpot: " << enableSpot;
    if (currentSpot) {

    } else {

    }
}

void SettingProxy::enablePoint(const bool &enablePoint) {
    if (enablePoint == currentPoint) return;
    currentPoint = enablePoint;
    qDebug() << "enablePoint: " << enablePoint;
    if (currentPoint) {

    } else {

    }
}

void SettingProxy::setPitch(const float &pitch) {
    qDebug() << "Pitch: " << pitch;

}

void SettingProxy::setYaw(const float &yaw) {
    qDebug() << "Yaw: " << yaw;

}