//
// Created by 冬榆 on 2026/1/17.
//

#include "SkyBox.hpp"

#include <memory>
#include "Mesh.hpp"
#include "Shape.hpp"

SkyBox::SkyBox() : mesh(std::make_shared<Mesh>()){
    // 生成2个三角形面网格
    SubMesh subMesh(this->mesh);
    subMesh.setMaterial(std::make_shared<Material>());
    mesh->addSubMesh(subMesh);
    Vertex v1;
    Vertex v2;
    Vertex v3;
    Vertex v4;
    v1.position = {-1, -1, 1};
    v2.position = {1, -1, 1};
    v3.position = {-1, 1, 1};
    v4.position = {1, 1, 1};
    this->mesh->addVertex(v1);
    this->mesh->addVertex(v2);
    this->mesh->addVertex(v3);
    this->mesh->addVertex(v4);
    this->mesh->addTri(0,1,2);
    this->mesh->addTri(1,3,2);
    (*mesh)[0].updateCount(mesh);
}

std::shared_ptr<Mesh> SkyBox::getMesh() const{
    return this->mesh;
}