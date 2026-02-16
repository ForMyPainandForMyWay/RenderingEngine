//
// Created by 冬榆 on 2026/1/17.
//

#include "SkyShader.hpp"
#include "F2P.hpp"
#include "LerpTool.hpp"
#include "Shape.hpp"
#include "Uniform.hpp"

SkyShader* SkyShader::shader = nullptr;

SkyShader* SkyShader::GetInstance() {
    if (shader == nullptr) shader = new SkyShader();
    return shader;
}

V2F SkyShader::VertexShader(
    const Vertex &vex,
    const Uniform &u,
    const GlobalUniform &gu) {
    V2F result;
    result.clipPosi = vex.getHomoPosi();
    result.invW = 1.0f/result.clipPosi[3];

    // 这里的MVP实际是R^{T}P^{-1}
    auto tmp = u.MVP * result.clipPosi;
    result.worldPosi = {tmp[0], tmp[1], tmp[2]};  // viewRay
    result.worldPosi = normalize(result.worldPosi);
    return result;
}

void SkyShader::GeometryShader(
    Triangle &tri,
    const std::shared_ptr<Material> &material,
    const std::array<Lights, 3> &PixLight,
    const std::vector<Lights> &VexLight,
    const MainLight *mainLight,
    const std::shared_ptr<ShadowMap> &shadowMap,
    const EnvironmentLight *envlight,
    const GlobalUniform &gu) {}

inline float smoothstep(const float a, const float b, const float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

F2P SkyShader::FragmentShader(
    const Fragment &frag,
    const std::shared_ptr<Material> &material,
    const std::array<Lights, 3> &light,
    const MainLight *mainLight,
    const std::shared_ptr<ShadowMap> &shadowMap,
    const EnvironmentLight *envlight,
    const GlobalUniform &gu,
    bool NeedShadow) {
    F2P pix;
    pix.keep();
    pix.x = frag.x;
    pix.y = frag.y;
    // 根据高度插值天空颜色
    const float h = std::clamp(frag.worldPosi[1], 0.0f, 1.0f);
    constexpr Vec3 skyTop = {0.30f, 0.65f, 1.0f};
    constexpr Vec3 skyHorizon = {0.9f,  0.9f,  1.0f};
    Vec3 color = lerp(skyHorizon, skyTop, h);
    const float t = smoothstep(-0.5f, 0.0f, frag.worldPosi[1]);
    color *= t;
    // Pixel 需要uint8_t类型的颜色值
    pix.Albedo = FloatPixel(color[0], color[1], color[2]);
    pix.depth = 1.0f;   // 天空永远最远
    return pix;
}