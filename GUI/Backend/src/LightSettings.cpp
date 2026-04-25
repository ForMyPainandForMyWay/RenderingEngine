//
// Created by 冬榆 on 2026/2/15.
//

#include <QDebug>

#include "FrameProvider.hpp"
#include "IEngine.hpp"
#include "SettingProxy.hpp"

// 灯光设置
void SettingProxy::enableEnv(const bool &enableEnv) {
    if (enableEnv == currentEnv) return;
    currentEnv = enableEnv;
    if (currentEnv) {
        engine->SetEnvLight(envCorlor[0], envCorlor[1], envCorlor[2], 1.0f);
    } else {
        engine->SetEnvLight(0, 0, 0, 5.0f);
    }
}

void SettingProxy::enableSpot(const bool &enableSpot) {
    if (enableSpot == currentSpot) return;
    currentSpot = enableSpot;
    if (currentSpot) {
        engine->SetMainLight(spotColor[0], spotColor[1], spotColor[2], 5.0f);
    } else {
        engine->SetMainLight(0, 0, 0, 0.0f);
    }
}

void SettingProxy::enablePoint(const bool &enablePoint) {
    if (enablePoint == currentPoint) return;
    currentPoint = enablePoint;
    if (currentPoint) {
        engine->SetPixLight(PixL1, pointColor[0], pointColor[1], pointColor[2], 5.0f);
    } else {
        engine->SetPixLight(PixL1, 0, 0, 0, 1.0f);
    }
}

void SettingProxy::setPitch(const float &pitch) {
    if (objID == -1) return;
    const float delta = currentPitch - pitch;
    currentPitch = pitch;
    MarkSceneChanged();
    engine->addTfCommand(objID, RenderObject, ROTATE, {0, delta*180, 0});
}

void SettingProxy::setYaw(const float &yaw) {
    if (objID == -1) return;
    const float delta = currentYaw - yaw;
    currentYaw = yaw;
    MarkSceneChanged();
    engine->addTfCommand(objID, RenderObject, ROTATE, {delta*180, 0, 0});
}

void SettingProxy::setEnvColor(const float &r, const float &g, const float &b) {
    envCorlor[0] = static_cast<uint8_t>(r * 255);
    envCorlor[1] = static_cast<uint8_t>(g * 255);
    envCorlor[2] = static_cast<uint8_t>(b * 255);
    if (currentEnv) {
        engine->SetEnvLight(envCorlor[0], envCorlor[1], envCorlor[2], 1.0f);
    }
}

void SettingProxy::setSpotColor(const float &r, const float &g, const float &b) {
    spotColor[0] = static_cast<uint8_t>(r * 255);
    spotColor[1] = static_cast<uint8_t>(g * 255);
    spotColor[2] = static_cast<uint8_t>(b * 255);
    if (currentSpot) {
        engine->SetMainLight(spotColor[0], spotColor[1], spotColor[2], 5.0f);
    }
}

void SettingProxy::setPointColor(const float &r, const float &g, const float &b) {
    pointColor[0] = static_cast<uint8_t>(r * 255);
    pointColor[1] = static_cast<uint8_t>(g * 255);
    pointColor[2] = static_cast<uint8_t>(b * 255);
    if (currentPoint) {
        engine->SetPixLight(PixL1, pointColor[0], pointColor[1], pointColor[2], 5.0f);
    }
}