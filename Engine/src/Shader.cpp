//
// Created by 冬榆 on 2025/12/30.
//

#include "Shader.h"
#include "F2P.h"
#include "FragTool.h"
#include "Mesh.h"
#include "Uniform.h"


Shader* Shader::shader = nullptr;

Shader* Shader::GetInstance() {
    if (shader == nullptr) shader = new Shader();
    return shader;
}

V2F Shader::VertexShader(const Vertex &vex, const Uniform &u) {
    auto clip = u.MVP * vex.getHomoIndex();
    // 注意这里的法线与uv提前乘了一个w (clip[3])
    // 注意取消提前乘了，clip空间裁剪插值不用透视
    const auto normal = u.normalTfMat * vex.getHomoNormal();
    return {clip, normal, vex.uv, 1/clip[3]};
}

void Shader::setMaterial(Material *mat) {
    material = mat;
}

F2P Shader::FragmentShader(const Fragment &frag, const Uniform &uni) const {
    F2P pix;
    pix.keep();
    pix.x = frag.x;
    pix.y = frag.y;
    // 输出片段着色器输入信息
    if (material == nullptr || material->KdMap == nullptr) {
        // 当没有材质时，填充默认颜色
        pix.color.r = 100; pix.color.g = 100; pix.color.b = 100;
        pix.color.a = 100;
    return pix;
    }
    // 纹理采样
    pix.color = BilinearSample(frag.uv, material->KdMap);
    // pix.color = Sample(frag.uv, material->KdMap);
    // TODO: 光照计算
    // TODO: mipmap->各向异性过滤
    return pix;
}
