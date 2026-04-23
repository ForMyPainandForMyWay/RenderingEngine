//
// Created by 冬榆 on 2026/2/15.
//

#include <QDebug>
#include <QUrl>
#include <thread>

#include "FrameProvider.hpp"
#include "IEngine.hpp"
#include "SettingProxy.hpp"


void SettingProxy::setRenderMode(const bool &isRaster) {
    if (isRaster == currentMode) return;
    currentMode = isRaster;
    if (!currentMode) {
        engine->SetRtMode();
    } else {
        engine->SetRasMode();
    }
}

void SettingProxy::setFXAA(const int &FXAALevel) {
    if (FXAALevel == currentFXAA) return;
    currentFXAA = FXAALevel;
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
    if (currentSky) {
        engine->OpenSky();
    } else {
        engine->CloseSky();
    }
}

void SettingProxy::enableSSAO(const bool &enableSSAO) {
    if (enableSSAO == currentSSAO) return;
    currentSSAO = enableSSAO;
    if (currentSSAO) {
        engine->OpenAO();
    } else {
        engine->CloseAO();
    }
}

void SettingProxy::enableShadow(const bool &enableShadow) {
    if (enableShadow == currentShadow) return;
    currentShadow = enableShadow;
    if (currentShadow) {
        engine->OpenShadow();
    } else {
        engine->CloseShadow();
    }
}

void SettingProxy::openObj(const QUrl &url) {
    const std::string filepath = url.toLocalFile().toUtf8().toStdString();
    // 注意这里读取完毕后，需要更新一下统计数据再发送信号
    engine->stopLoop();
    const auto meshId = engine->addMesh(filepath);
    uint16_t objID_ = engine->addObjects(meshId[0]);

    delete currentRecv;
    if (fp) {
        currentRecv = new Receiver(fp, 400, 400);
        engine->startLoop({objID_}, currentRecv);
    }
    objID = objID_;
    const auto [tri, vex] = engine->getTriVexNums();
    triNums = static_cast<int>(tri);
    vexNums = static_cast<int>(vex);

    emit TriangleNumsChanged();
    emit VertexNumsChanged();
}

void SettingProxy::saveImg(const QUrl &url) const {
    if (!fp) return;
    // 设置需要保存帧的标志
    fp->m_needSaveFrame = true;
    // 等待一帧数据更新
    std::this_thread::sleep_for(std::chrono::milliseconds(16)); // 等待约一帧的时间
    if (const QImage image = fp->getCurrentFrame();
        !image.isNull()) {
        if (const QString filePath = url.toLocalFile();
            image.save(filePath)) {
            qDebug() << "保存成功";
        } else qDebug() << "保存失败";
        }
}