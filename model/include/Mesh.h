//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_MESH_H
#define UNTITLED_MESH_H

#include <string>
#include <vector>
#include "Vec.hpp"
#include "Shape.h"

class Film;
class Shader;
class Mesh;
struct Material;
struct TextureMap;


// 表面材质
struct Material {
    std::string name;
    VecN<3> Ka;   // 环境光
    VecN<3> Kd;   // 漫反射
    VecN<3> Ks;   // 高光
    float Ns{};     // 光泽指数
    std::string map_Kd; // 纹理贴图名字(注：材质实际不止会有一个mak_Kd,需要拓展)

    Material();

    std::vector<Shader*> shaders;  // 延迟渲染,0阴影渲染,1光照渲染
    [[nodiscard]] Shader* getShader(const int pass) const
        { return shaders[pass]; };
};

// 纹理贴图
struct TextureMap {
    int width{}, height{};
    // std::string map_Kd;  // 贴图名字
    Film *uvImg;             // 贴图数组

    explicit TextureMap(const std::string& path);  // 从文件初始化贴图胶片
    ~TextureMap();
};

// 子网络模型,指定Mesh中应用某材质的三角面区间(以顶点为单位)
class SubMesh {
public:
    explicit SubMesh(const Mesh* mesh);
    void setMaterial(Material *mat);
    void updateCount(const Mesh* mesh);
    [[nodiscard]] uint32_t getOffset() const;
    [[nodiscard]] uint32_t getIdxCount() const;
    [[nodiscard]] bool vexIsEmpty() const;
    [[nodiscard]] bool triIsEmpty() const;
    [[nodiscard]] bool materialIsEmpty() const;
    [[nodiscard]] std::string getMaterialName() const;
    [[nodiscard]] Material* getMaterial() const;

    friend class Mesh;

protected:
    // 需要渲染的三角面的顶点区间索引,offset所指即为首个顶点
    uint32_t indexOffset{}, indexCount{};
    Material* material;  // 材质,默认为nullptr

};

// 将子网络模型组装为一个Mesh，存储顶点以及三角面顶点索引
class Mesh {
public:
    [[nodiscard]] size_t getSubMeshNums() const;
    void addSubMesh(SubMesh& submesh);
    [[nodiscard]] bool subIsEmpty() const;
    void setName(const std::string &name);
    std::string getName();

    SubMesh& operator[](size_t index);
    SubMesh const& operator[](size_t index) const;

    void addTri(uint32_t p1, uint32_t p2, uint32_t p3);
    void addVertex(Vertex vex);
    [[nodiscard]] size_t getVBONums() const;
    [[nodiscard]] size_t getTriNums() const;
    [[nodiscard]] bool vexIsEmpty() const;
    [[nodiscard]] bool triIsEmpty() const;


    auto begin()
        { return subMeshes.begin(); }
    auto end()
        { return subMeshes.end(); }
    [[nodiscard]] auto begin() const
        { return subMeshes.begin(); }
    [[nodiscard]] auto end()   const
        { return subMeshes.end(); }
    [[nodiscard]] auto cbegin() const
        { return subMeshes.cbegin(); }
    [[nodiscard]] auto cend()   const
        { return subMeshes.cend(); }

    [[nodiscard]] auto VexBegin() const { return VBO.cbegin(); }
    [[nodiscard]] auto VexMEnd()   const { return VBO.cend(); }
    [[nodiscard]] const SubMesh& getSubMesh(const size_t index) const { return subMeshes[index]; }

    friend class Graphic;

protected:
    std::vector<Vertex> VBO;   // 渲染顶点表
    std::vector<uint32_t> EBO;  // 三角面顶点索引，三个一组
    std::vector<SubMesh> subMeshes;
    std::string MeshName;
};


#endif //UNTITLED_MESH_H