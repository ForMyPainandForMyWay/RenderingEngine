//
// Created by yyd on 2025/12/24.
//

#include "ModelReader.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>



// 解析材质文件
void ModelReader::readMTLFile(const std::string &mtlFilename,
                 std::unordered_map<std::string, Material*> &materialMap) {
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
            currentMat = new Material();
            currentMat->name = matName;
            materialMap[matName] = currentMat;
        } else if (currentMat) {
            if (prefix == "Ka") ss >> currentMat->Ka[0] >> currentMat->Ka[1] >> currentMat->Ka[2];
            else if (prefix == "Kd") ss >> currentMat->Kd[0] >> currentMat->Kd[1] >> currentMat->Kd[2];
            else if (prefix == "Ks") ss >> currentMat->Ks[0] >> currentMat->Ks[1] >> currentMat->Ks[2];
            else if (prefix == "Ns") ss >> currentMat->Ns;
            else if (prefix == "map_Kd") ss >> currentMat->map_Kd;
        }
    }
}



// 读取模型obj文件，处理顶点、平面，再整合为若干模型输出
void ModelReader::readModelFile(const std::string &filename,
                                std::vector<Mesh> &meshes,
                                std::unordered_map<std::string, Material*> &materialMap) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return;
    }
    std::filesystem::path p(filename);
    std::string parent_path = p.parent_path().string();

    std::vector<VecN<3>> positions;
    std::vector<VecN<3>> normals;
    std::vector<VecN<2>> uvs;
    Mesh currentMesh;
    std::string line;

    auto pushCurrentMesh = [&]() {
        if (!currentMesh.triangles.empty() || !currentMesh.vertices.empty()) {
            meshes.emplace_back(currentMesh);
            currentMesh = Mesh();
        }
    };

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v") {
            VecN<3> pos{};
            ss >> pos[0] >> pos[1] >> pos[2];
            positions.emplace_back(pos);
            currentMesh.vertices.emplace_back();
        }
        else if (prefix == "vn") {
            VecN<3> n{};
            ss >> n[0] >> n[1] >> n[3];
            normals.emplace_back(n);
        }
        else if (prefix == "vt") {
            VecN<2> uv{};
            ss >> uv[0] >> uv[1];
            uv[1] = 1.0f - uv[1];  // 这里是为了blender反转
            uvs.emplace_back(uv);
        }
        else if (prefix == "f") {
            std::vector<Vertex> faceVerts;
            std::string vertexStr;
            while (ss >> vertexStr) {
                int vIdx = 0, vtIdx = 0, vnIdx = 0;
                // 支持 v/vt/vn 或 v//vn 或 v/vt
                size_t first = vertexStr.find('/');
                size_t last = vertexStr.rfind('/');
                if (first == std::string::npos) {
                    vIdx = std::stoi(vertexStr);
                } else if (first == last) {
                    vIdx = std::stoi(vertexStr.substr(0, first));
                    vtIdx = std::stoi(vertexStr.substr(first + 1));
                } else {
                    vIdx = std::stoi(vertexStr.substr(0, first));
                    vtIdx = std::stoi(vertexStr.substr(first + 1, last - first - 1));
                    vnIdx = std::stoi(vertexStr.substr(last + 1));
                }

                Vertex vert{};
                if (vIdx) vert.position = positions[vIdx - 1];
                if (vtIdx) vert.uv = uvs[vtIdx - 1];
                if (vnIdx) vert.normal = normals[vnIdx - 1];
                faceVerts.emplace_back(vert);
            }
            processPolygon(faceVerts, currentMesh.triangles);
        }
        else if (prefix == "mtllib") {
            std::string mtlFileName_;
            ss >> mtlFileName_;
            std::filesystem::path mtlPath = std::filesystem::path(parent_path) / mtlFileName_;
            std::string mtlFileName = mtlPath.string();
            readMTLFile(mtlFileName, materialMap);
        }
        else if (prefix == "usemtl") {
            std::string matName;
            ss >> matName;
            if (auto it = materialMap.find(matName); it != materialMap.end())
                currentMesh.material = it->second;
            else
                currentMesh.material = nullptr; // 找不到材质则用默认
        }
        else if (prefix == "o") {
            pushCurrentMesh();
        }
    }
    pushCurrentMesh();
}
