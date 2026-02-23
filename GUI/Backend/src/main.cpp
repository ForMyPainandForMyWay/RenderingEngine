#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QUrl>
#include <QQmlContext>  // 添加这一行
#include <QQmlEngine>
#include <QDir>

#include "FrameProvider.hpp"
#include "IEngine.hpp"
#include "SettingProxy.hpp"

int main(int argc, char *argv[]) {
    // 设置高 DPI 缩放（Qt 5.6+ 推荐）
    const QGuiApplication app(argc, argv);
    IEngine* Rengine = CreateEngine(400, 400, false, false);

    auto rengine = std::unique_ptr<IEngine>(Rengine);
    rengine->SetEnvLight(100, 100, 100, 1.0f);
    rengine->SetMainLight(0, 0, 0, 1.0f);
    const auto pLight = rengine->addPixLight(0, 0, 0, Point);
    rengine->addTfCommand(0, CameraID, TRANSLATE, {0.0f, 0.0f, 5.0f});
    rengine->addTfCommand(1, MainLightID, TRANSLATE, {0.0f, 0.0f, 5.0f});
    rengine->addTfCommand(pLight, PixL1, TRANSLATE, {0.0f, 0.0f, 5.0f});

    FrameProvider provider;
    SettingProxy controller((std::move(rengine)), &provider);

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("settingProxy", &controller);
    engine.rootContext()->setContextProperty("frameProvider", &provider);

    // 加载 QML
    const QUrl url(QStringLiteral("qrc:/qml/MainWindow.qml"));
    // 加载渲染引擎

    // 错误处理：加载失败时退出
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
        &app, [url](const QObject *obj, const QUrl &objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        }, Qt::QueuedConnection);
    engine.load(url);

    return QGuiApplication::exec();
}
