//
// Created by 冬榆 on 2026/1/17.
//

#ifndef RENDERINGENGINE_SHADER_H
#define RENDERINGENGINE_SHADER_H
#include "V2F.hpp"


class EnvironmentLight;
struct ShadowMap;
class MainLight;
class Lights;
struct Triangle;
struct Material;
struct Fragment;
struct Vertex;
class GlobalUniform;
struct Uniform;

class Shader {
public:
    virtual ~Shader() = default;

    virtual V2F VertexShader(
        const Vertex &vex,
        const Uniform &u,
        const GlobalUniform &gu) = 0;
    virtual void GeometryShader(
        Triangle &tri,
        const std::shared_ptr<Material> &material,
        const std::array<Lights, 3> &PixLight,
        const std::vector<Lights> &VexLight,
        const MainLight *mainLight,
        const std::shared_ptr<ShadowMap> &shadowMap,
        const EnvironmentLight *envlight,
        const GlobalUniform &gu) = 0;
    virtual F2P FragmentShader(
        const Fragment &frag,
        const std::shared_ptr<Material> &material,
        const std::array<Lights, 3> &light,
        const MainLight *mainLight,
        const std::shared_ptr<ShadowMap> &shadowMap,
        const EnvironmentLight *envlight,
        const GlobalUniform &gu,
        bool NeedShadow) = 0;
};


#endif //RENDERINGENGINE_SHADER_H