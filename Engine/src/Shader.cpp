//
// Created by 冬榆 on 2025/12/30.
//

#include "Shader.h"
#include "V2F.h"
#include "Uniform.h"
#include "Shape.h"

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