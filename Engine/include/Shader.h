//
// Created by 冬榆 on 2025/12/30.
//

#ifndef UNTITLED_SHADER_H
#define UNTITLED_SHADER_H

class Uniform;
struct V2F;
struct Vertex;


class Shader {
public:
    static Shader* GetInstance();
    void SetUniform(Uniform *ufm) { this->uniform = ufm; }

    [[nodiscard]] V2F VertexShader(const Vertex &vex) const;

protected:
    Uniform *uniform = nullptr;
    static Shader *shader;
};


#endif //UNTITLED_SHADER_H