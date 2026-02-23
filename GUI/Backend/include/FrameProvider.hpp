//
// Created by 冬榆 on 2026/2/22.
//

#ifndef RENDERGUI_FRAMEPROVIDER_HPP
#define RENDERGUI_FRAMEPROVIDER_HPP
#include <QVideoSink>
#include <QVideoFrame>
#include <QMutex>
#include <QPointer>

#include "IEngine.hpp"


class FrameProvider : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVideoSink* videoSink READ videoSink WRITE setVideoSink NOTIFY videoSinkChanged)

public:
    explicit FrameProvider(QObject *parent = nullptr);

    [[nodiscard]] QVideoSink* videoSink() const;
    void setVideoSink(QVideoSink* newVideoSink);
    void updateTexture(const uchar* data, int width, int height) ;

    signals:
    void videoSinkChanged();

private:
    QPointer<QVideoSink> m_videoSink;
    QMutex m_mutex; // 互斥锁
};

class Receiver: public IFrameReceiver {
public:
    Receiver(FrameProvider* fp, const int w, const int h) : m_frameProvider(fp), m_width(w), m_height(h){}
    Receiver() = default;
    void OnFrameReady(const void* data) override;

    FrameProvider* m_frameProvider = nullptr;  // 不拥有，只引用
    int m_width{}, m_height{};
};

#endif //RENDERGUI_FRAMEPROVIDER_HPP