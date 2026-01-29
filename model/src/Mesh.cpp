//
// Created by yyd on 2025/12/24.
//

#include "Mesh.hpp"
#include "UVLoader.hpp"
#include "BlinnShader.hpp"
#include "SkyShader.hpp"


Material::Material() : shaders(3, nullptr) {
    shaders[0] = SkyShader::GetInstance();
    shaders[1] = BlinnShader::GetInstance();
    shaders[2] = BlinnShader::GetInstance();
}

void Material::setKdTexture(const std::shared_ptr<TextureMap> &kd) {
    KdMap = kd;
}

// 从图像初始化纹理贴图数据
TextureMap::TextureMap(const std::string &path) {
    this->uvImg = loadPNG(path);
    this->width = this->uvImg->width;
    this->height = this->uvImg->height;
}

// 注意需要在Mesh添加三角形之前调用
SubMesh::SubMesh(const std::shared_ptr<Mesh>& mesh) {
    this->material = nullptr;
    this->indexCount = 0;
    this->indexOffset = mesh->getVBONums();  // 使用mesh的长度作为起始偏移量
}

// 用于更新subMesh的游标范围，需要传入Mesh指针
void SubMesh::updateCount(const std::shared_ptr<Mesh>& mesh) {
    if (mesh->getEBONums() == this->indexOffset) this->indexCount = 0;
    else this->indexCount = mesh->getEBONums() - this->indexOffset;
}

uint32_t SubMesh::getOffset() const {
    return this->indexOffset;
}

// 返回顶点数量
uint32_t SubMesh::getIdxCount() const {
    return this->indexCount;
}

bool SubMesh::vexIsEmpty() const {
    return indexCount==0;
}
bool SubMesh::triIsEmpty() const {
    return indexCount==0;
}

bool SubMesh::materialIsEmpty() const {
    return material==nullptr;
}

std::string SubMesh::getMaterialName() const {
    if (!materialIsEmpty())return this->material->map_Kd;
    return "No Material";
}

std::shared_ptr<Material> SubMesh::getMaterial() const {
    return material;
}

void SubMesh::setMaterial(const std::shared_ptr<Material> &mat) {
    this->material = mat;
}


// 添加三角形，参数为三个顶点的索引
void Mesh::addTri(uint32_t p1, uint32_t p2, uint32_t p3) {
    if (p1 >= this->VBO.size() ||
        p2 >= this->VBO.size() ||
        p3 >= this->VBO.size()) return;
    this->EBO.emplace_back(p1);
    this->EBO.emplace_back(p2);
    this->EBO.emplace_back(p3);
}

// 添加顶点,不保证去重，需要外部保证
void Mesh::addVertex(Vertex vex) {
    this->VBO.emplace_back(vex);
}




size_t Mesh::getSubMeshNums() const {
    return this->subMeshes.size();
}

void Mesh::addSubMesh(SubMesh& submesh) {
    this->subMeshes.emplace_back(submesh);
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

size_t Mesh::getVBONums() const {
    return this->VBO.size();
}

size_t Mesh::getEBONums() const {
    return this->EBO.size();
}

// 返回三角形数量
size_t Mesh::getTriNums() const {
    return this->EBO.size() / 3;
}

bool Mesh::vexIsEmpty() const {
    return this->VBO.empty();
}

bool Mesh::triIsEmpty() const {
    return this->EBO.empty();
}