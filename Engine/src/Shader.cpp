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

V2F Shader::VertexShader(const Vertex &vex) const {
    auto clip = uniform->MVP * vex.getHomoIndex();
    // 注意这里的法线与uv提前乘了一个w
    const auto normal = uniform->normalTfMat * vex.getHomoNormal() * clip[3];
    return {clip, normal, vex.uv * clip[3], 1/clip[3]};
}