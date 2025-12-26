//
// Created by yyd on 2025/12/24.
//

#include "ModelReader.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "Mesh.h"


// 解析材质文件,存入哈希表
void ModelReader::readMTLFile(const std::string &mtlFilename,
                              std::unordered_map<std::string, Material*> &materialMap,
                              std::unordered_map<std::string, TextureMap*> &textureMap) {
    std::ifstream mtlFile(mtlFilename);
    if (!mtlFile.is_open()) {
        std::cerr << "Cannot open MTL file: " << mtlFilename << "\n";
        return;
    }

    Material* currentMat = nullptr;
    std::string line;
    while (std::getline(mtlFile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "newmtl") {
            std::string matName;
            ss >> matName;
            // 查找哈希表,如果没有该元素，则存入哈希表
            if ( materialMap.find(matName) == materialMap.end()) {
                currentMat = new Material();
                currentMat->name = matName;
                materialMap[matName] = currentMat;
            }
        } else if (currentMat) {
            if (prefix == "Ka") ss >> currentMat->Ka[0] >> currentMat->Ka[1] >> currentMat->Ka[2];
            else if (prefix == "Kd") ss >> currentMat->Kd[0] >> currentMat->Kd[1] >> currentMat->Kd[2];
            else if (prefix == "Ks") ss >> currentMat->Ks[0] >> currentMat->Ks[1] >> currentMat->Ks[2];
            else if (prefix == "Ns") ss >> currentMat->Ns;
            else if (prefix == "map_Kd") {
                ss >> currentMat->map_Kd;  // 记录纹理贴图名字（路径）
                // 查找哈希表,如果没有该元素，则存入哈希表
                if (textureMap.find(currentMat->map_Kd) == textureMap.end()) {
                    textureMap[currentMat->map_Kd] = new TextureMap(currentMat->map_Kd);
                }
            }
        }
    }
}



/*
  读取模型obj文件，处理顶点、平面，再整合为若干模型输出，连带读取MTL文件与纹理
  注意传入的Mesh表、材质表、uv表
*/
void ModelReader::readObjFile(
    const std::string& filename,
    std::vector<Mesh*>& meshes,
    std::unordered_map<std::string, Material*>& materialMap,
    std::unordered_map<std::string, TextureMap*>& textureMap)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open Obj file: " << filename << "\n";
        return;
    }

    std::filesystem::path p(filename);
    std::string parent_path = p.parent_path().string();

    std::vector<VecN<3>> positions;
    std::vector<VecN<3>> normals;
    std::vector<VecN<2>> uvs;

    auto currentMesh = new Mesh();
    auto currentSubMesh = new SubMesh();

    auto pushSubMesh = [&]() {
        if (!currentSubMesh->triIsEmpty()) {
            currentMesh->addSubMesh(currentSubMesh);
            currentSubMesh = new SubMesh();
        }
    };

    auto pushMesh = [&]() {
        pushSubMesh();
        if (!currentMesh->subIsEmpty()) {
            meshes.emplace_back(currentMesh);
            currentMesh = new Mesh();
        }
    };

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v") {
            VecN<3> pos{};
            ss >> pos[0] >> pos[1] >> pos[2];
            positions.emplace_back(pos);
        }
        else if (prefix == "vn") {
            VecN<3> n{};
            ss >> n[0] >> n[1] >> n[2];
            normals.emplace_back(n);
        }
        else if (prefix == "vt") {
            VecN<2> uv{};
            ss >> uv[0] >> uv[1];
            uv[1] = 1.0f - uv[1];
            uvs.emplace_back(uv);
        }
        else if (prefix == "f") {
            std::vector<Vertex> faceVerts;
            std::string vertexStr;

            while (ss >> vertexStr) {
                int vIdx = 0, vtIdx = 0, vnIdx = 0;
                size_t first = vertexStr.find('/');
                size_t last  = vertexStr.rfind('/');

                if (first == std::string::npos) {
                    vIdx = std::stoi(vertexStr);
                } else if (first == last) {
                    vIdx  = std::stoi(vertexStr.substr(0, first));
                    vtIdx = std::stoi(vertexStr.substr(first + 1));
                } else {
                    vIdx  = std::stoi(vertexStr.substr(0, first));
                    vtIdx = std::stoi(vertexStr.substr(first + 1, last - first - 1));
                    vnIdx = std::stoi(vertexStr.substr(last + 1));
                }

                Vertex vert{};
                if (vIdx)  vert.position = positions[vIdx - 1];
                if (vtIdx) vert.uv       = uvs[vtIdx - 1];
                if (vnIdx) vert.normal   = normals[vnIdx - 1];

                faceVerts.emplace_back(vert);
            }

            currentSubMesh->Poly2Tri(faceVerts);
        }
        else if (prefix == "mtllib") {
            std::string mtl;
            ss >> mtl;
            readMTLFile((std::filesystem::path(parent_path) / mtl).string(),
                        materialMap, textureMap);
        }
        else if (prefix == "usemtl") {
            pushSubMesh();

            std::string matName;
            ss >> matName;
            auto it = materialMap.find(matName);
            currentSubMesh->setMaterial(it != materialMap.end() ? it->second : nullptr);
        }
        else if (prefix == "o" || prefix == "g") {
            pushMesh();
        }
    }

    pushMesh(); // 文件结束
}
