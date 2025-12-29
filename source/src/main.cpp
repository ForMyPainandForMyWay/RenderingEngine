#include <iostream>

#include "Mat.hpp"
#include "Mesh.h"
#include "ModelReader.h"
#include "thread_pool.h"

class SimpleTask : public Task{
    void run() override {
        std::cout << "test\n" << std::endl;
    }
};

int main() {
    std::unordered_map<std::string, Material*> materialMap;
    std::unordered_map<std::string, TextureMap*> textureMap;
    std::unordered_map<std::string, Mesh*> meshes;
    const std::string filename = R"(/Users/dongyu/CLionProjects/RenderEngine/bin/test.obj)";

    ModelReader::readObjFile(filename, meshes, materialMap, textureMap);
    std::cout << "mat " << materialMap.size() << std::endl;
    std::cout << "mesh " << meshes.size() << std::endl;

    // 输出所有Material信息
    std::cout << "\n=== Material Information ===" << std::endl;
    for (const auto& [name, material] : materialMap) {
        std::cout << "Material Name: " << name << std::endl;
        std::cout << "  Ka (Ambient): [" << material->Ka[0] << ", " << material->Ka[1] << ", " << material->Ka[2] << "]" << std::endl;
        std::cout << "  Kd (Diffuse): [" << material->Kd[0] << ", " << material->Kd[1] << ", " << material->Kd[2] << "]" << std::endl;
        std::cout << "  Ks (Specular): [" << material->Ks[0] << ", " << material->Ks[1] << ", " << material->Ks[2] << "]" << std::endl;
        std::cout << "  Ns (Shininess): " << material->Ns << std::endl;
        std::cout << "  Texture Map: " << material->map_Kd << std::endl;
        std::cout << std::endl;
    }

    // 输出所有TextureMap信息
    std::cout << "\n=== TextureMap Information ===" << std::endl;
    for (const auto& [name, texture] : textureMap) {
        std::cout << "Texture Name: " << name << std::endl;
        std::cout << "  Width: " << texture->width << std::endl;
        std::cout << "  Height: " << texture->height << std::endl;
        std::cout << "  Image Pointer: " << texture->uvImg << std::endl;
        std::cout << std::endl;
    }

    // 输出所有Mesh信息
    std::cout << "\n=== Mesh Information ===" << std::endl;
    for (const auto& [name, mesh] : meshes) {
        std::cout << "Mesh Name: " << name << std::endl;
        std::cout << "  SubMesh Count: " << mesh->getSubMeshNums() << std::endl;

        for (size_t i = 0; i < mesh->getSubMeshNums(); ++i) {
            const SubMesh& subMesh = (*mesh)[i];
            std::cout << "    SubMesh " << i << ":" << std::endl;
            std::cout << "      Vertex Count: " << subMesh.getVexNums() << std::endl;
            std::cout << "      Triangle Index Count: " << subMesh.getVexNums() << std::endl;
            std::cout << "      Material: " << subMesh.getMaterialName() << std::endl;
        }
        std::cout << std::endl;
    }

    // MatMN<4, 4>a;
    // MatMN<4, 4>b;
    // MatMN<4, 4>c;
    // ThreadPool thread_pool{};
    // thread_pool.addTask(
        // new FuncTask(
        // [&a, &b, &c]() {
            // c=a+b;
        // }
        // )
    // );
    // thread_pool.wait();

    return 0;
}