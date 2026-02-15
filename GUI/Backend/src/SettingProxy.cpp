//
// Created by 冬榆 on 2026/2/15.
//

#include "SettingProxy.hpp"

SettingProxy::SettingProxy(QObject *parent)
    : QObject(parent){}
SettingProxy::~SettingProxy() = default;


int SettingProxy::TriangleNums() const {
    return triNums;
}

int SettingProxy::VertexNums() const {
    return vexNums;
}