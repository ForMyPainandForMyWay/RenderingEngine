#include <array>
#include <atomic>
#include <iostream>
#include <string>

#include "Film.hpp"
#include "IEngine.hpp"

class Reciver: public IFrameReceiver {
public:
    static std::atomic<size_t> i;
    Reciver() = default;
    ~Reciver() override = default;
    void OnFrameReady(const void* data) override {
        Film film(800, 800);
        film.copyFromPtr(static_cast<const unsigned char*>(data));
        std::string filename = "./test/" +
            std::to_string(i) + ".pam";
        film.save(filename);  // 转为 const char*
        ++i;
        std::cout << "保存第 " << i-1 << " 张图片" << std::endl;
    }
};
std::atomic<size_t> Reciver::i = 0;

void testRt() {
    IEngine* engine = CreateEngine(800, 800, true, true);

    const auto meshName = R"(./test4.obj)";
    const auto meshId = engine->addMesh(meshName);
    uint16_t objID = engine->addObjects(meshId[0]);

    // 灯光
    const auto meshName3 = R"(./Light.obj)";
    const auto meshId3 = engine->addMesh(meshName3);
    const uint16_t objID3 = engine->addObjects(meshId3[0]);

    // 天花板
    const uint16_t objID8 = engine->addObjects(engine->addMesh(R"(./Cube.obj)")[0]);

    // 墙壁
    const auto meshName4 = R"(./test.obj)";
    const auto meshId4 = engine->addMesh(meshName4);
    const uint16_t objID4 = engine->addObjects(engine->addMesh(R"(./Cube.obj)")[0]);  // 地板
    const uint16_t objID5 = engine->addObjects(meshId4[0]);
    const uint16_t objID6 = engine->addObjects(meshId4[0]);
    const uint16_t objID7 = engine->addObjects(meshId4[0]);

    const auto [t, v] = engine->getTriVexNums();
    std::cout << "三角形数量: " << t << " 顶点数量: " << v << std::endl;

    // 苏珊娜
    engine->addTfCommand(objID, RenderObject, SCALE, {0.7f, 0.7f, 0.7f});
    engine->addTfCommand(objID, RenderObject, ROTATE, {-45.0f, 0.0f, 0.0f});
    engine->addTfCommand(objID, RenderObject, TRANSLATE, {0.0f, -0.5f, 0.0f});

    // 墙壁
    engine->addTfCommand(objID3, RenderObject, TfType::TRANSLATE, {0.0f, 2.0f,0.0f});
    engine->addTfCommand(objID3, RenderObject, TfType::SCALE, {0.5f, 1.f,0.5f});
    engine->addTfCommand(objID4, RenderObject, TfType::TRANSLATE, {0.0f,-2.0f,0.0f});
    engine->addTfCommand(objID5, RenderObject, TfType::TRANSLATE, {2.0f, 0.0f,0.0f});
    engine->addTfCommand(objID5, RenderObject, TfType::ROTATE, {0.0f, 90.0f,0.0f});
    engine->addTfCommand(objID6, RenderObject, TfType::TRANSLATE, {-2.0f,0.f,0.0f});
    engine->addTfCommand(objID7, RenderObject, TfType::TRANSLATE, {0.0f,0.f,-2.0f});
    // 天花板
    engine->addTfCommand(objID8, RenderObject, TfType::TRANSLATE, {0.0f, 2.0f,0.0f});
    engine->addTfCommand(objID8, RenderObject, TfType::ROTATE, {0.0f, 180.0f,0.0f});

    // 相机灯光转动
    engine->addTfCommand(0, CameraID, TRANSLATE, {0.0f, 0.0f, 3.0f});

    engine->startLoop({ objID, objID3, objID4, objID5, objID6, objID7, }, new Reciver());
    while (true) {
        if (Reciver::i > 0) {
            engine->stopLoop();
            break;
        }
    }
}


void testRas() {
    IEngine* engine = CreateEngine(800, 800, true, false);

    const auto meshName = R"(./test4.obj)";
    const auto meshId = engine->addMesh(meshName);
    uint16_t objID = engine->addObjects(meshId[0]);

    // 物品
    const auto meshName2 = R"(./Cube.obj)";
    const auto meshId2 = engine->addMesh(meshName2);
    const uint16_t objID2 = engine->addObjects(meshId2[0]);

    const auto [t, v] = engine->getTriVexNums();
    std::cout << "三角形数量: " << t << " 顶点数量: " << v << std::endl;

    // engine->SetEnvLight(100, 100, 100, 1.0f);
    engine->SetMainLight(255 ,255, 255, 1.0f);

    // 物体转动
    engine->addTfCommand(objID2, RenderObject, ROTATE, {45.0f, 135.0f, 0.0f});
    engine->addTfCommand(objID2, RenderObject, TRANSLATE, {0.0f, 0.0f, -3.0f});


    // 相机灯光转动
    engine->addTfCommand(0, CameraID, TRANSLATE, {4.0f, 0.0f, 0.0f});
    engine->addTfCommand(0, CameraID, ROTATE, {0.0f, 40.0f, 0.0f});
    engine->addTfCommand(0, CameraID, TRANSLATE, {0.0f, 0.0f, 3.0f});

    engine->addTfCommand(1, MainLightID, TRANSLATE, {0.0f, 0.0f, 3.0f});

    engine->OpenShadow();
    // auto start = std::chrono::high_resolution_clock::now();
    // engine->RenderFrame({ objID, objID2});
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "渲染耗时: " << duration.count() << " 微秒\n";

    // 点光源测试
    // const auto plight = engine->addPixLight(255, 0, 0, LType::Point);
    // engine->SetPixLight(PixL1, 0, 255, 0, 1.0f);
    // engine->addTfCommand(plight, PixL1, TRANSLATE, {0.0f, 0.0f, 4.0f});


    // engine->startLoop({ objID, objID2,}, new Reciver());
    // while (true) {
    //     if (Reciver::i > 0) {
    //         engine->stopLoop();
    //         break;
    //     }
    // }
    int batch = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < batch; i++) {
        engine->RenderFrame({ objID, objID2,});
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "平均渲染耗时: " << duration.count() /  batch << " 微秒\n";
    std::cout << "平均渲染耗时: " << duration.count() /  (batch*1000.0) << " 毫秒\n";
    std::cout << "平均渲染耗时: " << duration.count() /  (batch*1000000.0) << " 秒\n";
    std::cout << "平均帧率: " <<   (batch * 1000000.0 / duration.count()) << std::endl;
}

void testLoop() {
    IEngine* engine = CreateEngine(800, 800, false, false);

    const auto meshName = R"(./test4.obj)";
    const auto meshId = engine->addMesh(meshName);
    uint16_t objID = engine->addObjects(meshId[0]);

    // 物品
    const auto meshName2 = R"(./Cube.obj)";
    const auto meshId2 = engine->addMesh(meshName2);
    const uint16_t objID2 = engine->addObjects(meshId2[0]);

    const auto [t, v] = engine->getTriVexNums();
    // std::cout << "三角形数量: " << t << " 顶点数量: " << v << std::endl;

    engine->SetEnvLight(100, 100, 100, 1.0f);
    engine->SetMainLight(255, 255, 255, 1.0f);

    // 物体转动
    engine->addTfCommand(objID2, RenderObject, ROTATE, {45.0f, 135.0f, 0.0f});
    engine->addTfCommand(objID2, RenderObject, TRANSLATE, {0.0f, 0.0f, -3.0f});


    // 相机灯光转动
    engine->addTfCommand(0, CameraID, TRANSLATE, {4.0f, 0.0f, 0.0f});
    engine->addTfCommand(0, CameraID, ROTATE, {0.0f, 40.0f, 0.0f});

    // 平移
    engine->addTfCommand(0, CameraID, TRANSLATE, {0.0f, 0.0f, 3.0f});
    engine->addTfCommand(1, MainLightID, TRANSLATE, {0.0f, 0.0f, 4.0f});

    engine->OpenShadow();
    engine->startLoop({ objID, objID2}, new Reciver());
    while (true) {
        if (Reciver::i > 5) {
            engine->stopLoop();
            break;
        }
    }
}

void testLight() {
    IEngine* engine = CreateEngine(800, 800, false, false);

    const auto meshName = R"(./test4.obj)";
    const auto meshId = engine->addMesh(meshName);
    uint16_t objID = engine->addObjects(meshId[0]);

    engine->SetEnvLight(100, 100, 100, 1.0f);
    engine->SetMainLight(255 ,255, 255, 1.0f);
    // auto pl = engine->addPixLight(255,255, 255, LType::Point);
    // auto vl = engine->addVexLight(255, 255, 255, LType::Point);

    // 灯光
    const auto meshName3 = R"(./Light.obj)";
    const auto meshId3 = engine->addMesh(meshName3);
    const uint16_t objID3 = engine->addObjects(meshId3[0]);

    // 墙壁
    const auto meshName4 = R"(./test.obj)";
    const auto meshId4 = engine->addMesh(meshName4);
    const uint16_t objID4 = engine->addObjects(meshId4[0]);
    const uint16_t objID5 = engine->addObjects(meshId4[0]);
    const uint16_t objID6 = engine->addObjects(meshId4[0]);
    const uint16_t objID7 = engine->addObjects(meshId4[0]);

    const auto [t, v] = engine->getTriVexNums();
    std::cout << "三角形数量: " << t << " 顶点数量: " << v << std::endl;

    // 苏珊娜
    engine->addTfCommand(objID, RenderObject, SCALE, {0.7f, 0.7f, 0.7f});
    engine->addTfCommand(objID, RenderObject, ROTATE, {-45.0f, 0.0f, 0.0f});
    engine->addTfCommand(objID, RenderObject, TRANSLATE, {0.0f, -0.5f, 0.0f});

    // 墙壁
    engine->addTfCommand(objID3, RenderObject, TfType::TRANSLATE, {0.0f, 2.0f,0.0f});
    engine->addTfCommand(objID4, RenderObject, TfType::TRANSLATE, {0.0f,-2.0f,0.0f});
    engine->addTfCommand(objID5, RenderObject, TfType::TRANSLATE, {2.0f, 0.0f,0.0f});
    engine->addTfCommand(objID5, RenderObject, TfType::ROTATE, {0.0f, 90.0f,0.0f});
    engine->addTfCommand(objID6, RenderObject, TfType::TRANSLATE, {-2.0f,0.f,0.0f});
    engine->addTfCommand(objID7, RenderObject, TfType::TRANSLATE, {0.0f,0.f,-2.0f});

    // 相机转动
    engine->addTfCommand(0, CameraID, TRANSLATE, {0.0f, 0.0f, 3.0f});
    // engine->addTfCommand(0, CameraID, ROTATE, {0.0f, 0.0f, 0.0f});

    // 主光源
    engine->addTfCommand(1, MainLightID, TRANSLATE, {0.0f, 2.0f, 0.0f});
    engine->addTfCommand(1, MainLightID, ROTATE, {-90.0f, 0.0f, 0.0f});

    // PixLight测试
    // engine->addTfCommand(pl, PixL1, TRANSLATE, {0.0f, 2.0f, 0.0f});
    // VexLight测试
    // engine->addTfCommand(vl, VexLight, TRANSLATE, {0.0f, 2.0f, 0.0f});

    engine->OpenShadow();

    engine->startLoop({ objID, objID3, objID4, objID5, objID6, objID7}, new Reciver());
    while (true) {
        if (Reciver::i > 0) {
            engine->stopLoop();
            break;
        }
    }
}

void testSpeed() {
    IEngine* engine = CreateEngine(800, 800, false, true);

    const auto meshName = R"(./test4.obj)";
    const auto meshId = engine->addMesh(meshName);
    uint16_t objID = engine->addObjects(meshId[0]);

    // 灯光
    const auto meshName3 = R"(./Light.obj)";
    const auto meshId3 = engine->addMesh(meshName3);
    const uint16_t objID3 = engine->addObjects(meshId3[0]);

    // 墙壁
    const auto meshName4 = R"(./test.obj)";
    const auto meshId4 = engine->addMesh(meshName4);
    const uint16_t objID4 = engine->addObjects(meshId4[0]);
    const uint16_t objID5 = engine->addObjects(meshId4[0]);
    const uint16_t objID6 = engine->addObjects(meshId4[0]);
    const uint16_t objID7 = engine->addObjects(meshId4[0]);

    const auto [t, v] = engine->getTriVexNums();
    std::cout << "三角形数量: " << t << " 顶点数量: " << v << std::endl;

    // 苏珊娜
    engine->addTfCommand(objID, RenderObject, SCALE, {0.7f, 0.7f, 0.7f});
    engine->addTfCommand(objID, RenderObject, ROTATE, {-45.0f, 0.0f, 0.0f});
    engine->addTfCommand(objID, RenderObject, TRANSLATE, {0.0f, -0.5f, 0.0f});

    // 墙壁
    engine->addTfCommand(objID3, RenderObject, TfType::TRANSLATE, {0.0f, 2.0f,0.0f});
    engine->addTfCommand(objID4, RenderObject, TfType::TRANSLATE, {0.0f,-2.0f,0.0f});
    engine->addTfCommand(objID5, RenderObject, TfType::TRANSLATE, {2.0f, 0.0f,0.0f});
    engine->addTfCommand(objID6, RenderObject, TfType::TRANSLATE, {-2.0f,0.f,0.0f});
    engine->addTfCommand(objID7, RenderObject, TfType::TRANSLATE, {0.0f,0.f,-2.0f});

    // 相机灯光转动
    engine->addTfCommand(0, CameraID, TRANSLATE, {0.0f, 0.0f, 3.0f});
    engine->addTfCommand(1, MainLightID, TRANSLATE, {0.0f, 0.0f, 4.0f});

    int batch = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < batch; i++) {
            engine->RenderFrame({ objID, objID3, objID4, objID5, objID6, objID7});
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "平均渲染耗时: " << duration.count() /  batch << " 微秒\n";
    std::cout << "平均渲染耗时: " << duration.count() /  (batch*1000.0) << " 毫秒\n";
    std::cout << "平均渲染耗时: " << duration.count() /  (batch*1000000.0) << " 秒\n";
    std::cout << "平均帧率: " <<   (batch * 1000000.0 / duration.count()) << std::endl;
}

int main() {
    // testRt();
    testRas();
    // testLight();
    // testSpeed();
    return 0;
}

