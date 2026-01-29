//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_MESH_H
#define UNTITLED_MESH_H

#include "Shape.hpp"

struct BLAS;
class Shader;
struct Film;
class BlinnShader;
class Mesh;
struct Material;
struct TextureMap;


// 表面材质
struct Material {
    std::string name;
    Vec3 Ka{1.0f, 1.0f, 1.0f};   // 环境光
    Vec3 Kd{1.0f,1.0f,1.0f};   // 漫反射
    Vec3 Ks{1.0f,1.0f,1.0f};   // 高光
    Vec3 Ke{0.f, 0.f, 0.f};  // 自发光
    float Ns=32;   // 光泽指数
    std::string map_Kd="None"; // 纹理贴图名字(注：材质实际不止会有一个mak_Kd,需要拓展)
    std::string map_Normal="None";  // 法线贴图名字
    std::shared_ptr<TextureMap> KdMap;  // 纹理贴图指针
    std::shared_ptr<TextureMap> NormalMap;  // 法线贴图指针
    Material();

    std::vector<Shader*> shaders;  // 延迟渲染,0阴影渲染,1光照渲染
    [[nodiscard]] Shader* getShader(const int pass) const
        { return shaders[pass]; }
    void setKdTexture(const std::shared_ptr<TextureMap>& kd);
};

// 纹理贴图
struct TextureMap {
    uint32_t width{}, height{};
    std::unique_ptr<Film> uvImg;  // 贴图数组

    explicit TextureMap(const std::string& path);  // 从文件初始化贴图胶片
    ~TextureMap() = default;
};

// 子网络模型,指定Mesh中应用某材质的三角面区间(以顶点为单位)
class SubMesh {
public:
    explicit SubMesh(const std::shared_ptr<Mesh>& mesh);
    void setMaterial(const std::shared_ptr<Material> &mat);
    void updateCount(const std::shared_ptr<Mesh> &mesh);
    [[nodiscard]] uint32_t getOffset() const;
    [[nodiscard]] uint32_t getIdxCount() const;
    [[nodiscard]] bool vexIsEmpty() const;
    [[nodiscard]] bool triIsEmpty() const;
    [[nodiscard]] bool materialIsEmpty() const;
    [[nodiscard]] std::string getMaterialName() const;
    [[nodiscard]] std::shared_ptr<Material> getMaterial() const;

    friend class Mesh;

protected:
    // 需要渲染的三角面的顶点区间索引,offset所指即为首个顶点
    uint32_t indexOffset{}, indexCount{};
    std::shared_ptr<Material> material;  // 材质,默认为nullptr
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
    [[nodiscard]] size_t getEBONums() const;
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
    [[nodiscard]] Vertex& getVertex(const size_t index) { return VBO[index]; }

    friend class Graphic;
    friend struct BLAS;
    int BLASIdx = -1;  // 模型BVH索引,为-1说明没有建立BVH
    std::shared_ptr<BLAS> BuildBLAS();

protected:
    std::vector<Vertex> VBO;   // 渲染顶点表
    std::vector<uint32_t> EBO;  // 三角面顶点索引，三个一组连续存储
    std::vector<SubMesh> subMeshes;
    std::string MeshName;
};


#endif //UNTITLED_MESH_H