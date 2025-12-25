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
    static void readModelFile(const std::string &filename, std::vector<Mesh> &meshes,
                        std::unordered_map<std::string, Material*> &materialMap);
    static void readMTLFile(const std::string &mtlFilename,
                            std::unordered_map<std::string, Material*> &materialMap);
};


#endif //UNTITLED_MODELREADER_H