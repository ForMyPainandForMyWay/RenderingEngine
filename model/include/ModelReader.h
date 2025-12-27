//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_MODELREADER_H
#define UNTITLED_MODELREADER_H
#include <unordered_map>

#include "Mesh.h"
#include <vector>

class ModelReader {
public:
    static void readObjFile(const std::string &filename,
                            std::unordered_map<std::string, Mesh*>& meshes,
                            std::unordered_map<std::string, Material*> &materialMap,
                            std::unordered_map<std::string, TextureMap*> &textureMap);
    static void readMTLFile(const std::string &mtlFilename,
                            std::unordered_map<std::string, Material*> &materialMap,
                            std::unordered_map<std::string, TextureMap*> &textureMap);
};


#endif //UNTITLED_MODELREADER_H