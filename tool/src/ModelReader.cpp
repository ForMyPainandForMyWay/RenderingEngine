//
// Created by yyd on 2025/12/24.
//

#include <fstream>
#include <iostream>
#include <sstream>

#include "ModelReader.hpp"

#include "GammaTool.hpp"
#include "Mesh.hpp"


// 解析材质文件,存入哈希表
void ModelReader::readMTLFile(
    const bool Gamma,
    const std::string &mtlFilename,
    std::unordered_map<std::string, std::shared_ptr<Material>> &materialMap,
    std::unordered_map<std::string, std::shared_ptr<TextureMap>> &textureMap,
    std::unordered_map<std::string, std::shared_ptr<TextureMap>> &normalMap) {

    std::ifstream mtlFile(mtlFilename);
    if (!mtlFile.is_open()) {
        std::cerr << "Cannot open MTL file: " << mtlFilename << "\n";
        return;
    }

    auto currentMat = std::make_shared<Material>();
    std::string line;
    while (std::getline(mtlFile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "newmtl") {
            std::string matName;
            ss >> matName;
            // 查找哈希表,如果没有该元素,则存入哈希表
            if (!materialMap.contains(matName)) {
                currentMat = std::make_shared<Material>();
                currentMat->name = matName;
                materialMap[matName] = currentMat;
            }
        } else if (currentMat) {
            if (prefix == "Ka") ss >> currentMat->Ka[0] >> currentMat->Ka[1] >> currentMat->Ka[2];
            else if (prefix == "Kd") ss >> currentMat->Kd[0] >> currentMat->Kd[1] >> currentMat->Kd[2];
            else if (prefix == "Ks") ss >> currentMat->Ks[0] >> currentMat->Ks[1] >> currentMat->Ks[2];
            else if (prefix == "Ke") ss >> currentMat->Ke[0] >> currentMat->Ke[1] >> currentMat->Ke[2];
            else if (prefix == "Ns") ss >> currentMat->Ns;
            else if (prefix == "map_Kd") {
                ss >> currentMat->map_Kd;  // 记录纹理贴图名字（路径）
                // 查找哈希表,如果没有该元素,则存入哈希表
                if (!textureMap.contains(currentMat->map_Kd)) {
                    auto texture = std::make_shared<TextureMap>(currentMat->map_Kd);
                    // 初始化之后转换为float(自动清空8位数据)
                    texture->uvImg->Trans2FloatPixel();
                    textureMap[currentMat->map_Kd] = texture;
                    currentMat->setKdTexture(texture);  // 设置材质的纹理贴图
                    // gamma矫正
                    if (Gamma)
                        GammaCorrect(texture->uvImg);
                } else currentMat->KdMap = textureMap[currentMat->map_Kd];
            }
            else if (prefix == "norm") {
                ss >> currentMat->map_Normal;
                if (!normalMap.contains(currentMat->map_Normal)) {
                    auto texture = std::make_shared<TextureMap>(currentMat->map_Normal);
                    texture->uvImg->Trans2FloatPixel();
                    normalMap[currentMat->map_Normal] = texture;
                    currentMat->NormalMap = texture;  // 设置材质的法线贴图
                } else currentMat->NormalMap = normalMap[currentMat->map_Normal];
            }
        }
    }
}



/*
  读取模型obj文件,处理顶点、平面,再整合为若干模型输出,连带读取MTL文件与纹理
  注意传入的Mesh表、材质表、uv表
*/
std::vector<std::string> ModelReader::readObjFile(
    const bool Gamma,
    const std::string &filename,
    std::unordered_map<std::string, std::shared_ptr<Mesh>> &meshes,
    std::unordered_map<std::string, std::shared_ptr<Material>> &materialMap,
    std::unordered_map<std::string, std::shared_ptr<TextureMap>> &textureMap,
    std::unordered_map<std::string, std::shared_ptr<TextureMap>> &normalMap){

    std::vector<std::string> meshId;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open Obj file: " << filename << "\n";
        return meshId;
    }

    std::filesystem::path p(filename);
    std::string parent_path = p.parent_path().string();

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<VecN<2>> uvs;

    auto currentMesh = std::make_shared<Mesh>();
    SubMesh currentSubMesh{currentMesh};
    // 用于标记顶点的哈希表，key为顶点在f行的“a/b/c”
    std::unordered_map<std::string, uint32_t> vertexMap;

    auto pushSubMesh = [&]() {
        currentSubMesh.updateCount(currentMesh);
        if (currentSubMesh.getIdxCount() != 0) {
            currentMesh->addSubMesh(currentSubMesh);
            currentSubMesh = SubMesh(currentMesh);
        }
    };

    auto pushMesh = [&]() {
        pushSubMesh();
        if (!currentMesh->vexIsEmpty() && !meshes.contains(currentMesh->getName())) {
            meshes[currentMesh->getName()] = currentMesh;
            currentMesh = std::make_shared<Mesh>();
            vertexMap.clear();
        }
    };

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v") {
            Vec3 pos{};
            ss >> pos[0] >> pos[1] >> pos[2];
            positions.emplace_back(pos);
        }
        else if (prefix == "vn") {
            Vec3 n{};
            ss >> n[0] >> n[1] >> n[2];
            normals.emplace_back(n);
        }
        else if (prefix == "vt") {
            VecN<2> uv{};
            ss >> uv[0] >> uv[1];
            uv[1] = 1.0f - uv[1];  // 贴图数据顶部为0，反转一下
            uvs.emplace_back(uv);
        }
        else if (prefix == "f") {
            std::string vertexStr;
            ObjFace face;
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

                // 添加逻辑：根据三个Idx确定Mesh中是否已经存在这个点
                if (!vertexMap.contains(vertexStr)) {
                    Vertex vert;
                    if (vIdx)  vert.position = positions[vIdx - 1];
                    if (vtIdx) vert.uv       = uvs[vtIdx - 1];
                    if (vnIdx) vert.normal   = normals[vnIdx - 1];
                    currentMesh->addVertex(vert);
                    vertexMap[vertexStr] = currentMesh->getVBONums()-1;
                }
                face.addVexIndex(vertexMap[vertexStr]);
            }
            splitPoly2Tri(face, currentMesh);
        }
        else if (prefix == "mtllib") {
            std::string mtl;
            ss >> mtl;
            readMTLFile(Gamma, (std::filesystem::path(parent_path) / mtl).string(),
                        materialMap, textureMap, normalMap);
        }
        else if (prefix == "usemtl") {
            pushSubMesh();
            std::string matName;
            ss >> matName;
            auto it = materialMap.find(matName);
            currentSubMesh.setMaterial(it != materialMap.end() ? it->second : nullptr);
        }
        else if (prefix == "o" || prefix == "g") {
            std::string modelName;
            ss >> modelName;
            currentMesh->setName(modelName);
            pushMesh();
            meshId.emplace_back(modelName);
        }
    }
    pushMesh(); // 文件结束
    return meshId;
}

// 将多边形切分为若干三角形，并写入各自顶点索引到Mesh的indices中
void ModelReader::splitPoly2Tri(const ObjFace& face, const std::shared_ptr<Mesh>& mesh) {
    for (size_t i=2; i < face.vertexIndices.size(); i++) {
        mesh->addTri(face[0], face[i-1], face[i]);
    }
}

// Gamma矫正解码
void ModelReader::GammaCorrect(const std::unique_ptr<Film> &img) {
    for (auto& [r,g,b,i] : img->floatImg) {
        r = srgbToLinear(r);
        g = srgbToLinear(g);
        b = srgbToLinear(b);
    }
}
