//
// Created by 冬榆 on 2025/12/30.
//

#ifndef UNTITLED_SHADER_H
#define UNTITLED_SHADER_H

#include <vector>

#include "Shader.h"
#include "VecPro.hpp"

struct Triangle;
class EnvironmentLight;
class MainLight;
class Lights;
struct Pixel;
class GlobalUniform;
struct ShadowMap;
struct Uniform;
struct V2F;
struct F2P;
struct Vertex;
struct Fragment;
struct Material;


class BlinnShader : public Shader {
public:
    BlinnShader() = default;
    static BlinnShader* GetInstance();
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
        const ShadowMap &shadowMap,
        const EnvironmentLight *envlight,
        const GlobalUniform &gu) override;

    F2P FragmentShader(
        const Fragment &frag,
        const std::shared_ptr<Material> &material,
        const std::array<Lights, 3> &light,
        const MainLight *mainLight,
        const ShadowMap &shadowMap,
        const EnvironmentLight *envlight,
        const GlobalUniform &gu,
        bool NeedShadow) override;

    static float CalcHardShadow(
        const MainLight* mainLight,
        const Vec4 &worldPos,
        const Vec3 &normal,
        const GlobalUniform &gu,
        const ShadowMap &ShadowMap);

protected:
    static BlinnShader *shader;
};

#endif //UNTITLED_SHADER_H