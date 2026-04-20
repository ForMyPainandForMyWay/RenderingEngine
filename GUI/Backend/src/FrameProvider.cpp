//
// Created by 冬榆 on 2026/2/22.
//
#include <QDebug>
#include "FrameProvider.hpp"

#include <iostream>
#include <thread>


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
    frame.unmap();
    {
        QMutexLocker locker(&m_mutex);
        // VideoOutput销毁后，m_videoSink 自动置空
        if (m_videoSink) {
            m_videoSink->setVideoFrame(frame);
        }
    }
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
    lastReady = std::chrono::high_resolution_clock::now();
}