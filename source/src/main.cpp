#include <iostream>

#include "Mat.hpp"
#include "thread_pool.h"

class SimpleTask : public Task{
    void run() override {
        std::cout << "test\n" << std::endl;
    }
};

int main() {
    // std::unordered_map<std::string, Material*> materialMap;
    // std::unordered_map<std::string, TextureMap*> textureMap;
    // std::unordered_map<std::string, Mesh*> meshes;
    // const std::string filename = R"(/Users/dongyu/CLionProjects/RenderEngine/bin/test.obj)";
    //
    // ModelReader::readObjFile(filename, meshes, materialMap, textureMap);
    // std::cout << "mat " <<materialMap.size() << std::endl;
    // std::cout << "mesh " << meshes.size() << std::endl;
    MatMN<4, 4>a;
    MatMN<4, 4>b;
    MatMN<4, 4>c;

    ThreadPool thread_pool{};
    thread_pool.addTask(
        new FuncTask(
        [&a, &b, &c]() {
            c=a+b;
        }
        )
    );
    thread_pool.wait();

    return 0;
}