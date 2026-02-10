#include <iostream>
#include <unordered_map>
#include <string>

#include "Engine.hpp"
#include "Mesh.hpp"

void dumpMaterialInfo(const std::unordered_map<std::string, Material*>& materialMap) {
    std::cout << "\n=== 材质信息 ===\n";
    for (const auto& [name, mat] : materialMap) {
        if (!mat) continue;
        std::cout << "材质名称: " << name << "\n";
        std::cout << "  Ka: [" << mat->Ka[0] << ", " << mat->Ka[1] << ", " << mat->Ka[2] << "]\n";
        std::cout << "  Kd: [" << mat->Kd[0] << ", " << mat->Kd[1] << ", " << mat->Kd[2] << "]\n";
        std::cout << "  Ks: [" << mat->Ks[0] << ", " << mat->Ks[1] << ", " << mat->Ks[2] << "]\n";
        std::cout << "  Ns: " << mat->Ns << "\n";
        std::cout << "  map_Kd: " << mat->map_Kd << "\n";
    }
}

void dumpTextureInfo(const std::unordered_map<std::string, TextureMap*>& textureMap) {
    std::cout << "\n=== 纹理信息 ===\n";
    for (const auto& [name, tex] : textureMap) {
        if (!tex) continue;
        std::cout << "纹理名称: " << name << "\n";
        std::cout << "  尺寸: " << tex->width << " x " << tex->height << "\n";
        std::cout << "  uvImg 指针: " << tex->uvImg << "\n";
    }
}

void dumpMeshInfo(const std::unordered_map<std::string, Mesh*>& meshes) {
    std::cout << "\n=== 网格信息 ===\n";
    for (const auto& [name, mesh] : meshes) {
        if (!mesh) continue;
        std::cout << "网格名称: " << name << "\n";
        std::cout << "  顶点数量: " << mesh->getVBONums() << "\n";
        std::cout << "  三角形(顶点)数量: " << mesh->getTriNums() << "\n";
        std::cout << "  子网格数量: " << mesh->getSubMeshNums() << "\n";

        // 输出所有顶点
        std::cout << "  顶点列表:\n";
        size_t idx = 0;
        for (auto it = mesh->VexBegin(); it < mesh->VexMEnd(); ++it) {
            std::cout << "    [" << idx++ << "] Pos: (" << it->position[0] << ", " << it->position[1] << ", " << it->position[2] << ")";
            std::cout << " Normal: (" << it->normal[0] << ", " << it->normal[1] << ", " << it->normal[2] << ")";
            std::cout << " UV: (" << it->uv[0] << ", " << it->uv[1] << ")\n";
        }

        // 输出每个子网格信息
        for (size_t i = 0; i < mesh->getSubMeshNums(); ++i) {
            const SubMesh& sm = (*mesh)[i];
            std::cout << "  子网格 " << i << ":\n";
            std::cout << "    索引范围: " << sm.getOffset() << " - " << (sm.getOffset() + sm.getIdxCount() - 1) << "\n";
            std::cout << "    顶点数量: " << sm.getIdxCount() << std::endl;
            std::cout << "    材质: "
                      << (sm.materialIsEmpty() ? "无" : sm.getMaterialName()) << "\n";
        }
        std::cout << std::endl;
    }
}

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

