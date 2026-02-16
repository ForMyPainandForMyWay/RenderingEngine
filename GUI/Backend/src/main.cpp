#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QUrl>
#include <QQmlContext>  // 添加这一行
#include <QQmlEngine>
#include <QDir>

#include "IEngine.hpp"
#include "SettingProxy.hpp"

int main(int argc, char *argv[]) {
    // 设置高 DPI 缩放（Qt 5.6+ 推荐）
    const QGuiApplication app(argc, argv);
    SettingProxy controller;
    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("settingProxy", &controller);

    // 加载 QML
    const QUrl url(QStringLiteral("qrc:/qml/MainWindow.qml"));
    // 加载渲染引擎
    IEngine* Rengine = CreateEngine(800, 800, false, false);

    // 错误处理：加载失败时退出
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
        &app, [url](const QObject *obj, const QUrl &objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        }, Qt::QueuedConnection);
    engine.load(url);

    return QGuiApplication::exec();
}
