//
// Created by 冬榆 on 2026/2/15.
//

#include <cmath>

#include "FrameProvider.hpp"
#include "IEngine.hpp"
#include "SettingProxy.hpp"


void SettingProxy::setFOV(const float &FOV) const {
    MarkSceneChanged();
    engine->setCameraFov(FOV * 90);
}
void SettingProxy::setNear(const float &near) const {
    MarkSceneChanged();
    engine->setCameraNear(0.1f + near * 4.9f);
}
void SettingProxy::setFar(const float &far) const {
    MarkSceneChanged();
    // 使用指数映射让变化更平滑: 1.0 ~ 100.0
    engine->setCameraFar(std::pow(10.0f, 0.0f + far * 2.0f));
}