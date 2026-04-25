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

extern std::vector<uint16_t> envObjs;  // 环境模型
extern std::atomic<bool> EnvChangeed;  // 渲染场景改变
extern std::atomic<uint8_t> SSP;

class FrameProvider : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVideoSink* videoSink READ videoSink WRITE setVideoSink NOTIFY videoSinkChanged)

public:
    explicit FrameProvider(QObject *parent = nullptr);
    std::atomic<bool> m_needSaveFrame = false; // 标志是否需要保存帧数据

    [[nodiscard]] QVideoSink* videoSink() const;
    void setVideoSink(QVideoSink* newVideoSink);
    void updateTexture(const uchar* data, int width, int height) ;
    QImage getCurrentFrame();

    signals:
    void videoSinkChanged();

private:
    QPointer<QVideoSink> m_videoSink;
    QMutex m_mutex; // 互斥锁
    QImage m_currentFrame; // 存储当前帧
};

class Receiver: public IFrameReceiver {
public:
    Receiver(FrameProvider* fp, const int w, const int h) : m_frameProvider(fp), m_width(w), m_height(h){}
    Receiver() = default;
    void OnFrameReady(const void* data) override;

    FrameProvider* m_frameProvider = nullptr;  // 不拥有，只引用
    std::function<void(uint8_t)> m_sspCallback;  // 用于设置SSP的回调
private:
    int m_width{}, m_height{};
    std::chrono::time_point<std::chrono::high_resolution_clock> lastReady = std::chrono::high_resolution_clock::now();
};

inline void MarkSceneChanged() {
    SSP = 1;
    EnvChangeed = true;
}
#endif //RENDERGUI_FRAMEPROVIDER_HPP