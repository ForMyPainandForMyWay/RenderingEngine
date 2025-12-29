//
// Created by yyd on 2025/12/24.
//

#include "Mesh.h"
#include "UVLoader.h"


// 从图像初始化纹理贴图数据
TextureMap::TextureMap(const std::string &path) {
    this->uvImg = loadPNG(path);
}

TextureMap::~TextureMap() {
    delete this->uvImg;
}

SubMesh::SubMesh() {
    this->material = nullptr;
}

size_t SubMesh::getVexNums() const {
    return this->vertices.size();
}

size_t SubMesh::getTriNums() const {
    return this->indices.size() / 3;
}

bool SubMesh::vexIsEmpty() const {
    return this->vertices.empty();
}

bool SubMesh::triIsEmpty() const {
    return this->indices.empty();
}

void SubMesh::setMaterial(Material *mat) {
    this->material = mat;
}

// 添加三角形，参数为三个顶点的索引
void SubMesh::addTri(uint32_t p1, uint32_t p2, uint32_t p3) {
    this->indices.emplace_back(p1);
    this->indices.emplace_back(p2);
    this->indices.emplace_back(p3);
}

// 添加顶点,不保证去重，需要外部保证
void SubMesh::addVertex(Vertex vex) {
    this->vertices.emplace_back(vex);
}

std::string SubMesh::getMaterialName() const {
    return this->material->map_Kd;
}



size_t Mesh::getSubMeshNums() const {
    return this->subMeshes.size();
}

void Mesh::addSubMesh(SubMesh&& submesh) {
    this->subMeshes.emplace_back(std::move(submesh));
}

bool Mesh::subIsEmpty() const {
    return this->subMeshes.empty();
}

void Mesh::setName(const std::string &name) {
    this->MeshName = name;
}

std::string Mesh::getName() {
    return this->MeshName;
}

SubMesh& Mesh::operator[](const size_t index) {
    return this->subMeshes.at(index);
}

SubMesh const& Mesh::operator[](const size_t index) const {
    return this->subMeshes.at(index);
}