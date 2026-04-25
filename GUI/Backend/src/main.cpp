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
    IEngine* Rengine = CreateEngine(400, 400, true, false);

    // 初始化引擎，调整光照相机
    auto rengine = std::unique_ptr<IEngine>(Rengine);
    rengine->SetEnvLight(100, 100, 100, 1.0f);
    rengine->SetMainLight(0, 0, 0, 1.0f);
    const auto pLight = rengine->addPixLight(0, 0, 0, Point);
    rengine->addTfCommand(0, CameraID, TRANSLATE, {0.0f, 0.0f, 3.0f});
    // rengine->addTfCommand(1, MainLightID, TRANSLATE, {0.0f, 0.0f, 5.0f});
    rengine->addTfCommand(1, MainLightID, TRANSLATE, {0.0f, 2.0f, 0.0f});
    rengine->addTfCommand(1, MainLightID, ROTATE, {-90.0f, 0.0f, 0.0f});
    rengine->addTfCommand(pLight, PixL1, TRANSLATE, {0.0f, 0.0f, 5.0f});

    // 添加墙壁和天花板
    // 顶灯
    const auto meshName3 = R"(./model/Light.obj)";
    const auto meshId3 = rengine->addMesh(meshName3);
    const uint16_t objID3 = rengine->addObjects(meshId3[0]);

    // 天花板
    const uint16_t objID8 = rengine->addObjects(rengine->addMesh(R"(./model/Cube.obj)")[0]);

    // 墙壁
    const auto meshName4 = R"(./model/test.obj)";
    const auto meshId4 = rengine->addMesh(meshName4);
    const uint16_t objID4 = rengine->addObjects(meshId4[0]);
    const uint16_t objID5 = rengine->addObjects(meshId4[0]);
    const uint16_t objID6 = rengine->addObjects(meshId4[0]);
    const uint16_t objID7 = rengine->addObjects(meshId4[0]);

    // 调整墙壁天花板位置
    rengine->addTfCommand(objID3, RenderObject, TfType::TRANSLATE, {0.0f, 2.0f,0.0f});
    rengine->addTfCommand(objID3, RenderObject, TfType::SCALE, {0.5f, 1.f,0.5f});
    rengine->addTfCommand(objID4, RenderObject, TfType::TRANSLATE, {0.0f,-2.0f,0.0f});
    rengine->addTfCommand(objID5, RenderObject, TfType::TRANSLATE, {2.0f, 0.0f,0.0f});
    rengine->addTfCommand(objID5, RenderObject, TfType::ROTATE, {0.0f, 90.0f,0.0f});
    rengine->addTfCommand(objID6, RenderObject, TfType::TRANSLATE, {-2.0f,0.f,0.0f});
    rengine->addTfCommand(objID7, RenderObject, TfType::TRANSLATE, {0.0f,0.f,-2.0f});
    // 天花板
    rengine->addTfCommand(objID8, RenderObject, TfType::TRANSLATE, {0.0f, 2.0f,0.0f});
    rengine->addTfCommand(objID8, RenderObject, TfType::ROTATE, {0.0f, 180.0f,0.0f});
    // 打包场景资源
    envObjs = {objID3, objID4, objID5, objID6, objID7, objID8};

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
