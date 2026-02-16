//
// Created by 冬榆 on 2026/1/17.
//

#ifndef RENDERINGENGINE_SKYSHADER_H
#define RENDERINGENGINE_SKYSHADER_H
#include "Shader.hpp"

class GlobalUniform;
struct Material;
struct Fragment;
struct Uniform;
struct Vertex;


class SkyShader : public Shader{
public:
    SkyShader() = default;
    static SkyShader* GetInstance();
    V2F VertexShader(
        const Vertex &vex,
        const Uniform &u,
        const GlobalUniform &gu) override;

    void GeometryShader(
        Triangle &tri,
        const std::shared_ptr<Material> &material,
        const std::array<Lights, 3> &PixLight,
        const std::vector<Lights> &VexLight,
        const MainLight *mainLight,
        const std::shared_ptr<ShadowMap> &shadowMap,
        const EnvironmentLight *envlight,
        const GlobalUniform &gu) override;

    F2P FragmentShader(
        const Fragment &frag,
        const std::shared_ptr<Material> &material,
        const std::array<Lights, 3> &light,
        const MainLight *mainLight,
        const std::shared_ptr<ShadowMap> &shadowMap,
        const EnvironmentLight *envlight,
        const GlobalUniform &gu,
        bool NeedShadow) override;

protected:
    static SkyShader *shader;
};


#endif //RENDERINGENGINE_SKYSHADER_H