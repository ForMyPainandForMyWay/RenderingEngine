//
// Created by 冬榆 on 2025/12/30.
//

#ifndef UNTITLED_SHADER_H
#define UNTITLED_SHADER_H

#include <vector>
#include "Vec.hpp"

struct Triangle;
class EnvironmentLight;
class MainLight;
class Lights;
struct Pixel;
class GlobalUniform;
struct ShadowMap;
class Uniform;
struct V2F;
struct F2P;
struct Vertex;
struct Fragment;
struct Material;


class BlinnShader {
public:
    static BlinnShader* GetInstance();
    static V2F VertexShader(
        const Vertex &vex,
        const Uniform &u);

    static void GeometryShader(
        Triangle &tri,
        const Material *material,
        const std::array<Lights, 3> &PixLight,
        const std::vector<Lights> &VexLight,
        const MainLight *mainLight,
        const ShadowMap &shadowMap,
        const EnvironmentLight *envlight,
        const GlobalUniform &gu);

    static  F2P FragmentShader(
        const Fragment &frag,
        const Material *material,
        const std::array<Lights, 3> &light,
        const MainLight *mainLight,
        const ShadowMap &shadowMap,
        const EnvironmentLight *envlight,
        const GlobalUniform &gu,
        bool NeedShadow);

    static Pixel CalcLight(
        const F2P &f2p,
        float shadow,
        const std::array<Lights, 3> &lights,
        const MainLight *mainLight,
        const EnvironmentLight *envlight,
        const GlobalUniform &gu,
        const Material *material);

    static float CalcHardShadow(
        const MainLight* mainLight,
        const VecN<4> &worldPos,
        const VecN<3> &normal,
        const GlobalUniform &gu,
        const ShadowMap &ShadowMap);
    void setMaterial(Material* mat);

protected:
    static BlinnShader *shader;
    Material*material = nullptr;
};

#endif //UNTITLED_SHADER_H