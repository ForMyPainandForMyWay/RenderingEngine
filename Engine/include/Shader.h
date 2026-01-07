//
// Created by 冬榆 on 2025/12/30.
//

#ifndef UNTITLED_SHADER_H
#define UNTITLED_SHADER_H

struct TextureMap;
class Uniform;
struct V2F;
struct F2P;
struct Vertex;
struct Fragment;
struct Material;

class Shader {
public:
    static Shader* GetInstance();
    static V2F VertexShader(const Vertex &vex, const Uniform &u) ;
    F2P FragmentShader(const Fragment &frag, const Uniform &u) const;
    void setMaterial(Material* mat);

protected:
    static Shader *shader;
    Material*material = nullptr;
};

/* TODO:
 * 结合经验，可以做两个pass，第一个pass利用光源的vp矩阵生成光源的shadowmap
 * 然后第二个pass利用这些光源的shadowmap计算光照
 * 为了实现这个，做两个shader，第一个用于shadow，Vertex时候传入光源vp，frag直接返回或者返回无效pix
 * 然后第二个shader正常计算，注意还要计算一个光源空间坐标用于在阴影贴图采样
 *
 * 还有一种方案是两步pass，第一步生成片元信息GBuffer，第二步直接遍历GBuffer进行统一光照
 * 还可以加第三步生成阴影
*/

#endif //UNTITLED_SHADER_H