#include <iostream>
#include <unordered_map>

#include "thread_pool.h"
#include "Mesh.h"
#include "ModelReader.h"

class SimpleTask : public Task{
    void run() override {
        std::cout << "test\n" << std::endl;
    }
};

int main() {
    std::unordered_map<std::string, Material*> materialMap;
    std::unordered_map<std::string, TextureMap*> textureMap;
    std::vector<Mesh*> meshes;
    const std::string filename = R"(/Users/dongyu/CLionProjects/RenderEngine/bin/test.obj)";

    ModelReader::readObjFile(filename, meshes, materialMap, textureMap);
    std::cout << "mat " <<materialMap.size() << std::endl;
    std::cout << "mesh " << meshes.size() << std::endl;
    std::cout << "subMeshes " << meshes[0]->getSubMeshNums() << std::endl;
    std::cout << "vex " << (*meshes[0])[0]->getVexNums() << std::endl;
    std::cout << "tri " << (*meshes[0])[0]->getTriNums() << std::endl;

    return 0;
}