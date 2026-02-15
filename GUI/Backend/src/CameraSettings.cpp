//
// Created by 冬榆 on 2026/2/15.
//

#include <QDebug>

#include "SettingProxy.hpp"


void SettingProxy::setFOV(const float &FOV) {
    qDebug() << "FOV: " << FOV;

}
void SettingProxy::setNear(const float &near) {
    qDebug() << "Near: " << near;

}
void SettingProxy::setFar(const float &far) {
    qDebug() << "Far: " << far;

}