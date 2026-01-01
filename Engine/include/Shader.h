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
    static V2F VertexShader(const Vertex &vex, const Uniform &u) ;

protected:
    // Uniform *uniform = nullptr;
    static Shader *shader;
};


#endif //UNTITLED_SHADER_H