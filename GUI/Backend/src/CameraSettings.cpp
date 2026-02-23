//
// Created by 冬榆 on 2026/2/15.
//

#include <QDebug>

#include "IEngine.hpp"
#include "SettingProxy.hpp"


void SettingProxy::setFOV(const float &FOV) const {
    engine->setCameraFov(FOV * 100);
}
void SettingProxy::setNear(const float &near) const {
    engine->setCameraNear(near * 5);

}
void SettingProxy::setFar(const float &far) const {
    engine->setCameraFar(far * 100);
}