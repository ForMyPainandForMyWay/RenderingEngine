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
    return this->triangles.size();
}

bool SubMesh::vexIsEmpty() const {
    return this->vertices.empty();
}

bool SubMesh::triIsEmpty() const {
    return this->triangles.empty();
}

// 扇形分割，多边形分割为三角形
void SubMesh::Poly2Tri(const std::vector<Vertex> &inVerts){
    processPolygon(inVerts, this->triangles);
}

void SubMesh::setMaterial(Material *mat) {
    this->material = mat;
}

size_t Mesh::getSubMeshNums() const {
    return this->subMeshes.size();
}

void Mesh::addSubMesh(SubMesh *submesh) {
    this->subMeshes.emplace_back(submesh);
}

bool Mesh::subIsEmpty() const {
    return this->subMeshes.empty();
}

SubMesh*& Mesh::operator[](const size_t index) {
    return this->subMeshes.at(index);
}

SubMesh* const& Mesh::operator[](const size_t index) const {
    return this->subMeshes.at(index);
}