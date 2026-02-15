//
// Created by 冬榆 on 2026/2/15.
//

#include <QDebug>
#include <QUrl>

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

    } else if (currentFXAA == 2) {

    } else if (currentFXAA == 1) {

    } else {

    }
}

void SettingProxy::enableSkyBox(const bool &enableSky) {
    if (enableSky == currentSky) return;
    currentSky = enableSky;
    qDebug() << "enableSkyBox: " << currentSky;
    if (currentSky) {

    } else {

    }

}

void SettingProxy::enableSSAO(const bool &enableSSAO) {
    if (enableSSAO == currentSSAO) return;
    currentSSAO = enableSSAO;
    qDebug() << "enableSSAO: " << currentSSAO;
    if (currentSSAO) {

    } else {

    }
}

void SettingProxy::enableShadow(const bool &enableShadow) {
    if (enableShadow == currentShadow) return;
    currentShadow = enableShadow;
    qDebug() << "enableShadow: " << currentShadow;
    if (currentShadow) {

    } else {

    }
}

void SettingProxy::openObj(const QUrl &url) {
    const QString filepath = url.toLocalFile();
    qDebug() << "openObj: " << filepath;

    // 注意这里读取完毕后，需要更新一下统计数据再发送信号

    emit TriangleNumsChanged();
    emit VertexNumsChanged();
}

void SettingProxy::saveImg(const QUrl &url) {
    const QString filepath = url.toLocalFile();
    qDebug() << "openObj: " << filepath;
}