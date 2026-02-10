#include <iostream>
#include <string>

#include "Engine.hpp"

int main() {
    Engine engine(800, 800, true, true);
    // const auto meshName = R"(./test4.obj)";
    const auto meshName = R"(./Frog.obj)";
    const auto meshId = engine.addMesh(meshName);
    uint16_t objID = engine.addObjects(meshId[0]);

    // 物品
    const auto meshName2 = R"(./Cube.obj)";
    const auto meshId2 = engine.addMesh(meshName2);
    const uint16_t objID2 = engine.addObjects(meshId2[0]);

    // 灯光
    const auto meshName3 = R"(./Light.obj)";
    const auto meshId3 = engine.addMesh(meshName3);
    const uint16_t objID3 = engine.addObjects(meshId3[0]);

    // 墙壁
    const auto meshName4 = R"(./test.obj)";
    const auto meshId4 = engine.addMesh(meshName4);
    const uint16_t objID4 = engine.addObjects(meshId4[0]);
    const uint16_t objID5 = engine.addObjects(meshId4[0]);
    const uint16_t objID6 = engine.addObjects(meshId4[0]);
    const uint16_t objID7 = engine.addObjects(meshId4[0]);

    engine.SetEnvLight(100, 100, 100, 1.0f);
    engine.SetMainLight();

    // 物体转动
    // engine.addTfCommand({objID2, RenderObject, TfCmd::ROTATE, {45.0f, 135.0f, 0.0f}});
    // engine.addTfCommand({objID2, RenderObject, TfCmd::SCALE, {0.5f, 0.5f, 0.5f}});

    // 蛙蛙
    engine.addTfCommand({objID, RenderObject, TfCmd::SCALE, {0.08f, 0.08f, 0.08f}});
    engine.addTfCommand({objID, RenderObject, TfCmd::ROTATE, {90.0f, 0.0f, 180.0f}});
    engine.addTfCommand({objID, RenderObject, TfCmd::TRANSLATE, {0.1f, -0.5f, 0.0f}});

    // 苏珊娜
    // engine.addTfCommand({objID, RenderObject, TfCmd::SCALE, {0.7f, 0.7f, 0.7f}});
    // engine.addTfCommand({objID, RenderObject, TfCmd::ROTATE, {-45.0f, 0.0f, 0.0f}});
    // engine.addTfCommand({objID, RenderObject, TfCmd::TRANSLATE, {0.0f, -0.5f, 0.0f}});

    // engine.addTfCommand({objID, TfCmd::ROTATE, {0.0f, 0.0f, 0.0f}});
    // engine.addTfCommand({objID, RenderObject, TfCmd::ROTATE, {-35.0f, 35.0f, 0.0f}});

    // 墙壁
    engine.addTfCommand({objID3, RenderObject, TfCmd::TRANSLATE, {0.0f, 2.0f,0.0f}});
    engine.addTfCommand({objID4, RenderObject, TfCmd::TRANSLATE, {0.0f,-2.0f,0.0f}});
    engine.addTfCommand({objID5, RenderObject, TfCmd::TRANSLATE, {2.0f, 0.0f,0.0f}});
    engine.addTfCommand({objID6, RenderObject, TfCmd::TRANSLATE, {-2.0f,0.f,0.0f}});
    engine.addTfCommand({objID7, RenderObject, TfCmd::TRANSLATE, {0.0f,0.f,-2.0f}});

    // 相机灯光转动
    // engine.addTfCommand({0, CameraID, TfCmd::TRANSLATE, {4.0f, 0.0f, 0.0f}});
    // engine.addTfCommand({0, CameraID, TfCmd::ROTATE, {0.0f, 40.0f, 0.0f}});
    // engine.addTfCommand({1, TfCmd::TRANSLATE, {3.0f, 4.0f, 4.5f}});
    // engine.addTfCommand({1, TfCmd::ROTATE, {-35.0f, 35.0f, 0.0f}});

    // 平移
    engine.addTfCommand({0, CameraID, TfCmd::TRANSLATE, {0.0f, 0.0f, 3.0f}});
    engine.addTfCommand({1, MainLightID, TfCmd::TRANSLATE, {0.0f, 0.0f, 4.0f}});

    // engine.OpenShadow();
    // engine.OpenSky();
    auto start = std::chrono::high_resolution_clock::now();
    // engine.RenderFrame({ objID2, objID3, objID4, objID5, objID6, objID7});
    // engine.RenderFrame({ objID3, objID2, objID6});

    engine.RenderFrame({ objID, objID3, objID4, objID5, objID6, objID7});
    // engine.RenderFrame({ objID});
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "渲染耗时: " << duration.count() << " 微秒\n";
    return 0;
}

