//
// Created by 冬榆 on 2026/2/22.
//
#include <QDebug>
#include <iostream>
#include <thread>

#include "FrameProvider.hpp"

std::vector<uint16_t> envObjs;
std::atomic<bool> EnvChangeed = true;
std::atomic<uint8_t> SSP = 1;


FrameProvider::FrameProvider(QObject *parent) : QObject(parent) {}

QVideoSink* FrameProvider::videoSink() const {
    return m_videoSink;
}

void FrameProvider::setVideoSink(QVideoSink* newVideoSink) {
    QMutexLocker locker(&m_mutex); // 加锁保护
    if (m_videoSink == newVideoSink) return;
    m_videoSink = newVideoSink; // QPointer 会自动监测对象销毁
    emit videoSinkChanged();
}
void FrameProvider::updateTexture(const uchar *data, const int width, const int height) {
    // data 指针指向的内容随时会变
    const QSize size(width, height);
    const QVideoFrameFormat format(size, QVideoFrameFormat::Format_RGBA8888);
    QVideoFrame frame(format);

    if (!frame.map(QVideoFrame::WriteOnly)) return;

    // 拷贝数据
    const int stride = frame.bytesPerLine(0);
    uchar* bits = frame.bits(0);
    const int lineBytes = width * 4;

    if (!data) {
        frame.unmap();
        return;
    }

    if (stride == lineBytes) {
        memcpy(bits, data, lineBytes * height);
    } else {
        for (int y = 0; y < height; ++y) {
            memcpy(bits + y * stride, data + y * lineBytes, lineBytes);
        }
    }
    // 只在需要保存帧时才更新当前帧
    if (m_needSaveFrame) {
        m_currentFrame = QImage(bits, width, height, QImage::Format_RGBA8888).copy();
        m_needSaveFrame = false; // 重置标志
    }
    frame.unmap();
    {
        QMutexLocker locker(&m_mutex);
        // VideoOutput销毁后，m_videoSink 自动置空
        if (m_videoSink) {
            m_videoSink->setVideoFrame(frame);
        }
    }
}

// 获取最近的一帧
QImage FrameProvider::getCurrentFrame() {
    QMutexLocker locker(&m_mutex);
    return m_currentFrame;
}

void Receiver::OnFrameReady(const void *data) {
    if (m_frameProvider == nullptr) return;
    const auto frame = static_cast<const uchar*>(data);
    const auto deltaT = std::chrono::high_resolution_clock::now() - lastReady;
    m_frameProvider->updateTexture(frame, m_width, m_height);
    // 渲染当前帧时，如果间隔小于1/60s，则阻塞直至下一帧
    if (deltaT > std::chrono::milliseconds(1000/60)) {
        std::this_thread::sleep_for(deltaT);
    }
    // 若场景未发生变换，则每帧递增SSP
    if (!EnvChangeed && SSP < 128) {
        ++SSP;
    }
    // 清除场景变化标志
    EnvChangeed = false;
    m_sspCallback(SSP);
    lastReady = std::chrono::high_resolution_clock::now();
}