//
// Created by 冬榆 on 2025/12/30.
//

#include <ranges>

#include "BlinnShader.h"
#include "F2P.h"
#include "FragTool.h"
#include "Mesh.h"
#include "Uniform.h"
#include "ShadowMap.h"
#include "Lights.h"


BlinnShader* BlinnShader::shader = nullptr;

BlinnShader* BlinnShader::GetInstance() {
    if (shader == nullptr) shader = new BlinnShader();
    return shader;
}

V2F BlinnShader::VertexShader(const Vertex &vex, const Uniform &u) {
    auto world = u.M * vex.getHomoIndex();
    auto clip = u.MVP * vex.getHomoIndex();
    // 注意这里的法线与uv提前乘了一个w (clip[3])
    // 注意取消提前乘了，clip空间裁剪插值不用透视
    const auto normal = u.normalTfMat * vex.getHomoNormal();
    return {world, clip, normal, vex.uv, 1/clip[3]};
}

void BlinnShader::setMaterial(Material *mat) {
    material = mat;
}

Pixel CalcLightDebug2(
    const F2P &f2p,
    const float shadow,
    const std::unordered_map<uint16_t, Lights> &lights,
    const MainLight *mainLight,
    const EnvironmentLight *envlight,
    const GlobalUniform &gu,
    const Material *material) {

    // 可视化世界坐标 worldPos
    const auto P = f2p.worldPosi;  // 假设 F2P 中有 worldPos 成员
    // 定义坐标可视化范围（可根据场景调整）
    constexpr float RANGE = 10.0f;  // 假设场景大致在 [-10, 10] 范围内
    Pixel p;
    p.r = static_cast<uint8_t>(std::clamp((P[0] / RANGE * 0.5f + 0.5f), 0.0f, 1.0f) * 255.0f);
    p.g = static_cast<uint8_t>(std::clamp((P[1] / RANGE * 0.5f + 0.5f), 0.0f, 1.0f) * 255.0f);
    p.b = static_cast<uint8_t>(std::clamp((P[2] / RANGE * 0.5f + 0.5f), 0.0f, 1.0f) * 255.0f);
    p.a = 255;
    return p;
}


F2P BlinnShader::FragmentShader(
    const Fragment &frag,
    const Material *material,
    const std::unordered_map<uint16_t, Lights> &light,
    const MainLight *mainLight,
    const ShadowMap &shadowMap,
    const EnvironmentLight *envlight,
    const GlobalUniform &gu,
    const bool NeedShadow) {
    F2P pix;
    pix.keep();
    pix.x = frag.x;
    pix.y = frag.y;
    pix.normal = frag.normal;
    pix.worldPosi = {frag.worldPosi[0], frag.worldPosi[1], frag.worldPosi[2], frag.worldPosi[3]};
    // 输出片段着色器输入信息
    if (material == nullptr || material->KdMap == nullptr) {
        // 当没有材质时，填充默认颜色
        pix.Albedo.r = 100; pix.Albedo.g = 100; pix.Albedo.b = 100;
        pix.Albedo.a = 255;
        return pix;
    }
    // 纹理采样
    pix.Albedo = BilinearSample(frag.uv, material->KdMap);

    // 阴影计算
    float shadow = 1.0f;
    if (NeedShadow)
        shadow = CalcHardShadow(mainLight, frag.worldPosi, frag.normal, gu, shadowMap);
    // 光照计算
    pix.Albedo = CalcLight(
        pix,
        shadow,
        light,
        mainLight,
        envlight,
        gu,
        material);
    return pix;
}

Pixel BlinnShader::CalcLight(
    const F2P &f2p,
    const float shadow,
    const std::unordered_map<uint16_t, Lights> &lights,
    const MainLight *mainLight,
    const EnvironmentLight *envlight,
    const GlobalUniform &gu,
    const Material *material) {
    // 中间计算用 float 颜色，0~1
    VecN<3> finalColorF = {0.0f, 0.0f, 0.0f};
    const auto Albedo = f2p.Albedo.toFloat();
    const VecN<3> fragPos = {f2p.worldPosi[0], f2p.worldPosi[1], f2p.worldPosi[2]}; // 片元世界坐标
    const auto V = normalize(gu.getCameraPos() - fragPos);  // 视线方向
    const auto N = normalize(f2p.normal);  // 法线

    // 计算环境光
    if (envlight != nullptr) {
        VecN<3> ambientF = Hadamard(Albedo, material->Ka);
        const auto radiance = envlight->getColor().toFloat() * envlight->getI();
        ambientF = Hadamard(ambientF, radiance);
        finalColorF += ambientF;
    }

    // 计算主光源(带有阴影)
    if (mainLight != nullptr) {
        const auto Lvec = mainLight->getPosi() - fragPos;
        const auto L = normalize(Lvec);  // 光照方向
        const auto H = normalize(L + V);  // 半程向量
        const auto R = Lvec * Lvec;
        const auto radiance = mainLight->getColor().toFloat() * mainLight->getI() / R;
        // 漫反射 注意这里的着色多乘了材质吸收率
        auto diffF = Hadamard(Albedo, material->Kd * std::max(N*L, 0.0f));
        diffF = Hadamard(diffF, radiance);
        // 镜面反射，高光不乘 Albedo
        auto specF = material->Ks * std::pow(std::max(N*H, 0.0f), material->Ns);
        specF = Hadamard(specF, radiance);
        finalColorF += (diffF+specF)*shadow;  // 注意主光源乘阴影系数
    }

    // 计算其他光源
    for (const auto &light : lights | std::views::values) {
        const auto Lvec = light.getPosi() - fragPos;
        const auto L = normalize(Lvec);
        const auto H = normalize(L + V);
        const auto R = Lvec * Lvec;
        const auto radiance = light.getColor().toFloat() * light.getI() / R;
        VecN<3> diffF = Hadamard(Albedo, material->Kd * std::max(N*L, 0.0f));
        diffF = Hadamard(diffF, radiance);
        finalColorF += diffF;
        if (light.getLType() == Ambient) continue;  // 环境光没有镜面反射
        VecN<3> specF = material->Ks * std::pow(std::max(N*H, 0.0f), material->Ns);
        specF = Hadamard(specF, radiance);
        finalColorF += specF;
    }

    // clamp 并映射回 0~255 Pixel
    Pixel finalColor;
    finalColor.r = static_cast<uint8_t>(std::clamp(finalColorF[0], 0.0f, 1.0f) * 255.0f);
    finalColor.g = static_cast<uint8_t>(std::clamp(finalColorF[1], 0.0f, 1.0f) * 255.0f);
    finalColor.b = static_cast<uint8_t>(std::clamp(finalColorF[2], 0.0f, 1.0f) * 255.0f);
    finalColor.a = 255;

    return finalColor;
}

float BlinnShader::CalcHardShadow(
    const MainLight* mainLight,
    const VecN<4> &worldPos,
    const VecN<3> &normal,
    const GlobalUniform &gu,
    const ShadowMap &ShadowMap) {
    auto lightClip = gu.getShadowPV() * worldPos;
    auto projPos = lightClip / lightClip[3];  // 透视除法->NDC空间
    const auto U = projPos[0] * 0.5f + 0.5f;  // 透视除法->shadow uv
    const auto V = 1 - projPos[1] * 0.5f + 0.5f;
    if (U < 0.0f || U > 1.0f || V < 0.0f || V > 1.0f)
        return 1.0f;

    constexpr float biasConst = 0.001f;
    constexpr float biasSlope = 0.01f;
    const auto N = normalize(normal);  // 法线
    const VecN<3> fragPos = {worldPos[0], worldPos[1], worldPos[2]}; // 片元世界坐标
    const auto L = normalize(mainLight->getPosi() - fragPos);  // 光照方向
    const float bias = biasConst + biasSlope * (1.0f - dot(N, L));

    const auto Z = projPos[2];  // 片元在光源空间的深度(归一化)
    const float depth = ShadowMap.Sample(U, V);
    const float shadow = (Z - bias > depth) ? 0.0f : 1.0f;
    return shadow;
}