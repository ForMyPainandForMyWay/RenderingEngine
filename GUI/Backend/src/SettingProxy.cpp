//
// Created by 冬榆 on 2026/2/15.
//

#include "SettingProxy.hpp"
#include "FrameProvider.hpp"


SettingProxy::SettingProxy(std::unique_ptr<IEngine> engine, FrameProvider* fp, QObject *parent)
    : QObject(parent) { this->engine = std::move(engine); this->fp = fp;}

SettingProxy::~SettingProxy() {
    if (engine) {
        engine->stopLoop();  // 等线程退出
    }
    delete currentRecv;
    engine = nullptr;
    fp = nullptr;
}


int SettingProxy::TriangleNums() const {
    return triNums;
}

int SettingProxy::VertexNums() const {
    return vexNums;
}