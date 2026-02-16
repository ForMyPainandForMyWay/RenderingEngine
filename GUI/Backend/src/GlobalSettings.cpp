//
// Created by 冬榆 on 2026/2/15.
//

#include <QDebug>
#include <QUrl>

#include "IEngine.hpp"
#include "SettingProxy.hpp"


void SettingProxy::setRenderMode(const bool &isRaster) {
    if (isRaster == currentMode) return;
    currentMode = isRaster;
    qDebug() << "setRenderMode is Raster:" << currentMode;
    // 后续完善流程
}

void SettingProxy::setFXAA(const int &FXAALevel) {
    if (FXAALevel == currentFXAA) return;
    currentFXAA = FXAALevel;
    qDebug() << "set FXAA: " << currentFXAA;
    if (currentFXAA == 3) {
        engine->SetAA(FXAAQ);
    } else if (currentFXAA == 2) {
        engine->SetAA(FXAAC);
    } else if (currentFXAA == 1) {
        engine->SetAA(FXAA);
    } else {
        engine->SetAA(NOAA);
    }
}

void SettingProxy::enableSkyBox(const bool &enableSky) {
    if (enableSky == currentSky) return;
    currentSky = enableSky;
    qDebug() << "enableSkyBox: " << currentSky;
    if (currentSky) {
        engine->OpenSky();
    } else {
        engine->CloseSky();
    }

}

void SettingProxy::enableSSAO(const bool &enableSSAO) {
    if (enableSSAO == currentSSAO) return;
    currentSSAO = enableSSAO;
    qDebug() << "enableSSAO: " << currentSSAO;
    if (currentSSAO) {
        engine->OpenAO();
    } else {
        engine->CloseAO();
    }
}

void SettingProxy::enableShadow(const bool &enableShadow) {
    if (enableShadow == currentShadow) return;
    currentShadow = enableShadow;
    qDebug() << "enableShadow: " << currentShadow;
    if (currentShadow) {
        engine->OpenShadow();
    } else {
        engine->CloseShadow();
    }
}

void SettingProxy::openObj(const QUrl &url) {
    const std::string filepath = url.toLocalFile().toUtf8().toStdString();
    qDebug() << "openObj: " << filepath;

    // 注意这里读取完毕后，需要更新一下统计数据再发送信号
    const auto meshId = engine->addMesh(filepath);
    uint16_t objID = engine->addObjects(meshId[0]);  // TODO: 注意这里还没有计划GUI的渲染物体列表
    const auto [tri, vex] = engine->getTriVexNums();
    triNums = static_cast<int>(tri);
    vexNums = static_cast<int>(vex);

    emit TriangleNumsChanged();
    emit VertexNumsChanged();
}

void SettingProxy::saveImg(const QUrl &url) {
    const QString filepath = url.toLocalFile();
    qDebug() << "openObj: " << filepath;
}