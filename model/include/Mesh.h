//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_MESH_H
#define UNTITLED_MESH_H

#include "Shape.h"
#include "film.h"

struct Material;
struct TextureMap;
//class Mesh;

// 表面材质
struct Material {
    std::string name;
    VecN<3> Ka;   // 环境光
    VecN<3> Kd;   // 漫反射
    VecN<3> Ks;   // 高光
    float Ns;     // 光泽指数
    std::string map_Kd; // 纹理贴图名字
};

// 纹理贴图
struct TextureMap {
    int width{}, height{};
    // std::string map_Kd;  // 贴图名字
    Film *uvImg;             // 贴图数组

    explicit TextureMap(const std::string& path);  // 从文件初始化贴图胶片
    ~TextureMap();
};

// 子网络模型，存储顶点以及三角面顶点索引
class SubMesh {
public:
    SubMesh();
    [[nodiscard]] size_t getVexNums() const;
    [[nodiscard]] size_t getTriNums() const;
    [[nodiscard]] bool vexIsEmpty() const;
    [[nodiscard]] bool triIsEmpty() const;
    void setMaterial(Material *mat);
    void addTri(uint32_t p1, uint32_t p2, uint32_t p3);
    void addVertex(Vertex vex);
    [[nodiscard]] std::string getMaterialName() const;

private:
    std::vector<Vertex> vertices;  // 渲染顶点表
    std::vector<uint32_t> indices;   // 三角面顶点索引，三个一组
    Material* material;            // 材质,默认为nullptr

};

// 将子网络模型组装为一个Mesh
class Mesh {
public:
    [[nodiscard]] size_t getSubMeshNums() const;
    void addSubMesh(SubMesh&& submesh);
    [[nodiscard]] bool subIsEmpty() const;
    void setName(const std::string &name);
    std::string getName();

    SubMesh& operator[](size_t index);
    SubMesh const& operator[](size_t index) const;

private:
    std::vector<SubMesh> subMeshes;
    std::string MeshName;
};


#endif //UNTITLED_MESH_H