//
// Created by 冬榆 on 2025/12/30.
//

#ifndef UNTITLED_SHADER_H
#define UNTITLED_SHADER_H

#include <unordered_map>

#include "Vec.hpp"

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
    static V2F VertexShader(const Vertex &vex, const Uniform &u) ;

    static  F2P FragmentShader(
        const Fragment &frag,
        const Material *material,
        const std::unordered_map<uint16_t, Lights> &light,
        const MainLight *mainLight,
        const ShadowMap &shadowMap,
        const EnvironmentLight *envlight,
        const GlobalUniform &gu,
        bool NeedShadow);

    static Pixel CalcLight(
        const F2P &f2p,
        float shadow,
        const std::unordered_map<uint16_t, Lights> &lights,
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