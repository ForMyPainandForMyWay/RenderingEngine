#include <iostream>
#include <unordered_map>
#include <string>

#include "Engine.h"
#include "Mesh.h"

// 假设 VecN, Vertex, Mesh, SubMesh, Material, TextureMap, Film 已定义

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
    // std::unordered_map<std::string, Material*> materialMap;
    // std::unordered_map<std::string, TextureMap*> textureMap;
    // std::unordered_map<std::string, Mesh*> meshes;
    //
    // const std::string filename = R"(/Users/dongyu/CLionProjects/RenderEngine/bin/test.obj)";
    // ModelReader::readObjFile(filename, meshes, materialMap, textureMap);
    //
    // std::cout << "加载完成！\n";
    // std::cout << "材质数量: " << materialMap.size() << "\n";
    // std::cout << "网格数量: " << meshes.size() << "\n";
    // std::cout << "纹理数量: " << textureMap.size() << "\n";
    //
    // dumpMaterialInfo(materialMap);
    // dumpTextureInfo(textureMap);
    // dumpMeshInfo(meshes);
    //
    // // 清理内存
    // for (auto& [_, mat] : materialMap) delete mat;
    // for (auto& [_, tex] : textureMap) delete tex;
    // for (auto& [_, mesh] : meshes) delete mesh;
    //
    // std::cout << "内存清理完成。\n";
    //
    //
    Engine engine(800, 800);
    const auto meshName = R"(/Users/dongyu/CLionProjects/RenderEngine/bin/test.obj)";
    const auto meshId = engine.addMesh(meshName);
    uint16_t objID = engine.addObjects(meshId[0]);
    engine.addTfCommand({0, TfCmd::TRANSLATE, {0.0f, 0.0f, 6.0f}});
    engine.addTfCommand({0, TfCmd::ROTATE, {0.0f, 20.0f, 45.0f}});
    engine.RenderFrame({objID});

    return 0;
}

