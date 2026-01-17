//
// Created by 冬榆 on 2025/12/30.
//

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

V2F BlinnShader::VertexShader(
    const Vertex &vex,
    const Uniform &u,
    const GlobalUniform &gu) {
    auto world = u.M * vex.getHomoIndex();
    auto clip = u.MVP * vex.getHomoIndex();
    const auto normal = normalize(u.normalTfMat * vex.getHomoNormal());
    return {world, clip, normal, vex.uv, 1/clip[3]};
}

void BlinnShader::GeometryShader(
    Triangle &tri,
    const std::shared_ptr<Material> &material,
    const std::array<Lights, 3> &PixLight,
    const std::vector<Lights> &VexLight,
    const MainLight *mainLight,
    const ShadowMap &shadowMap,
    const EnvironmentLight *envlight,
    const GlobalUniform &gu) {
    // 对三角形顶点计算TBN矩阵并应用于光源相机的位置向量
    // TBN计算(世界空间下)
    const auto E1 = tri[1].worldPosi - tri[0].worldPosi;
    const auto E2 = tri[2].worldPosi - tri[0].worldPosi;
    const auto deltaU1 = tri[1].uv[0] - tri[0].uv[0];
    const auto deltaU2 = tri[2].uv[0] - tri[0].uv[0];
    const auto deltaV1 = tri[1].uv[1] - tri[0].uv[1];
    const auto deltaV2 = tri[2].uv[1] - tri[0].uv[1];
    const auto denom = (deltaU1*deltaV2 - deltaU2*deltaV1);
    auto tmp = (E1*deltaV2-E2*deltaV1)/denom;
    const float hand = (denom > 0.0f) ? 1.0f : -1.0f;
    const VecN<3> T_raw = {tmp[0], tmp[1], tmp[2]};
    for (int i = 0; i < 3; ++i) {
        auto N = tri[i].normal;  // 世界空间的法向量
        auto T = normalize(T_raw - N*dot(N, T_raw));
        auto B = cross(N, T) * hand;
        const MatMN<3,3> invTBN(T, B, N);  // 直接列向量按行存储，转为逆矩阵
        // 应用TBN矩阵到主光源位置
        const auto triPosi = VecN<3>{tri[i].worldPosi[0], tri[i].worldPosi[1], tri[i].worldPosi[2]};
        if (mainLight!=nullptr) {
            // tri[i].MainLightOri = invTBN * (mainLight->getPosi() - triPosi);
            tri[i].MainLightOri = normalize(invTBN * (mainLight->getPosi() - triPosi));
        }
        // 应用TBN矩阵到相机位置
        tri[i].CameraOri = normalize(invTBN * (gu.getCameraPos() - triPosi));
        // 应用TBN矩阵到各逐像素光源位置
        for (auto j = 0; j < PixLight.size(); ++j) {
            const auto &l = PixLight[j];
            if (!l.alive) continue;
            const auto lposi = invTBN * (l.getPosi() - triPosi);
            tri[i].PixLightOri[j] = lposi;
        }
        // 为顶点计算逐顶点光源光照(不考虑法线贴图与镜面光)
        tri[i].VexLightF = {0.0f, 0.0f, 0.0f};
        for (auto &l : VexLight) {
            if (!l.alive) continue;
            const auto Lvec = l.getPosi() - triPosi;
            const auto L = normalize(Lvec);
            const auto R = dot(Lvec, Lvec);
            const auto radiance = l.getColor().toFloat() * l.getI() / R;
            VecN<3> diffF = material->Kd * std::max(normalize(N)*L, 0.0f);
            diffF = Hadamard(diffF, radiance);
            tri[i].VexLightF += diffF;
        }
    }
}

// 通用光照计算：只关心 N, L, V, 距离等，不关心它们来自哪个空间
VecN<3> ApplyLighting(
    const VecN<3>& Albedo,
    const std::shared_ptr<Material> &material,
    const VecN<3>& N,          // 单位法线
    const VecN<3>& L,          // 单位光照方向（指向光源）
    const VecN<3>& V,          // 单位视线方向（指向相机）
    const float dist2,               // 光源到片元距离平方（用于衰减）
    const VecN<3>& lightColor, // 光源颜色 * 强度
    const float shadow = 1.0f) {
    VecN<3> result{0, 0, 0};
    const float NdotL = std::max(dot(N, L), 0.0f);
    if (NdotL <= 0.0f) return result;

    const auto radiance = lightColor / dist2;
    VecN<3> diffF = Hadamard(Albedo, material->Kd * NdotL);
    diffF = Hadamard(diffF, radiance);
    result += diffF;

    // 镜面（除环境光外都加）
    if (dot(material->Ks, material->Ks) > 0) {
        const VecN<3> H = normalize(L + V);
        const float NdotH = std::max(dot(N, H), 0.0f);
        VecN<3> specF = material->Ks * std::pow(NdotH, material->Ns);
        specF = Hadamard(specF, radiance);
        result += specF;
    }
    result *= shadow;
    return result;
}

F2P BlinnShader::FragmentShader(
    const Fragment &frag,
    const std::shared_ptr<Material> &material,
    const std::array<Lights, 3> &light,
    const MainLight *mainLight,
    const ShadowMap &shadowMap,
    const EnvironmentLight *envlight,
    const GlobalUniform &gu,
    const bool NeedShadow) {

    F2P pix;
    pix.keep();
    pix.x = frag.x;
    pix.y = frag.y;
    // 若无有效材质或漫反射贴图，返回默认灰色
    if (material == nullptr || material->KdMap == nullptr) {
        pix.Albedo = {100, 100, 100, 255};
        return pix;
    }
    // 采样漫反射贴图
    pix.Albedo = BilinearSample(frag.uv, material->KdMap);
    // 判断是否有法线贴图
    const bool hasBump = (material->BumpMap != nullptr);
    // 法线贴图采样与解码（自动处理空指针）
    const VecN<3> bumpSample = Sample(frag.uv, material->BumpMap).toFloat()*2.0f-1.0f;
    // 阴影计算
    float shadow = 1.0f;
    if (NeedShadow && mainLight != nullptr) {
        shadow = CalcHardShadow(mainLight, frag.worldPosi, frag.normal, gu, shadowMap);
    }
    // 片元世界位置（前三维）
    const VecN<3> fragPos = {frag.worldPosi[0], frag.worldPosi[1], frag.worldPosi[2]};
    // 计算法线N和视线V
    VecN<3> N, V;
    if (hasBump) {
        N = normalize(bumpSample);          // 切线空间法线（假设已正交化）
        V = normalize(frag.CameraOri);     // 切线空间视线方向（由 GS 提供）
    } else {
        N = normalize(frag.normal);         // 世界空间几何法线
        V = normalize(gu.getCameraPos() - fragPos); // 世界空间视线
    }
    // 初始化最终颜色：顶点光照为基础
    VecN<3> finalColorF = frag.VexLightF;
    const auto Albedo = pix.Albedo.toFloat();
    // 环境光（与方向无关）
    if (envlight != nullptr) {
        VecN<3> ambientF = Hadamard(Albedo, material->Ka);
        const VecN<3> radiance = envlight->getColor().toFloat() * envlight->getI();
        ambientF = Hadamard(ambientF, radiance);
        finalColorF += ambientF;
    }
    // 主光源
    if (mainLight != nullptr) {
        const VecN<3> Lvec_world = mainLight->getPosi() - fragPos;
        const float dist2 = dot(Lvec_world, Lvec_world);
        const VecN<3> lightColor = mainLight->getColor().toFloat() * mainLight->getI();
        VecN<3> L;
        if (hasBump) L = normalize(frag.MainLightOri); // 切线空间方向（GS 已转换）
        else L = normalize(Lvec_world);        // 世界空间方向
        finalColorF += ApplyLighting(
            Albedo, material, N, L, V, dist2, lightColor, shadow
        );
    }

    // 次要逐像素光源
    for (int i = 0; i < light.size(); ++i) {
        if (!light[i].alive || light[i].getLType() == Ambient) continue;

        const VecN<3> Lvec_world = light[i].getPosi() - fragPos;
        const float dist2 = dot(Lvec_world, Lvec_world);
        const VecN<3> lightColor = light[i].getColor().toFloat() * light[i].getI();

        VecN<3> L;
        if (hasBump) L = normalize(frag.PixLightOri[i]); // 切线空间方向
        else L = normalize(Lvec_world);          // 世界空间方向

        finalColorF += ApplyLighting(
            // 注意这里shadow直接赋值为1，不参与阴影计算
            Albedo, material, N, L, V, dist2, lightColor, 1.0f
        );
    }

    // 转换为 Pixel 输出
    pix.Albedo = {
        static_cast<uint8_t>(std::clamp(finalColorF[0], 0.0f, 1.0f) * 255.0f),
        static_cast<uint8_t>(std::clamp(finalColorF[1], 0.0f, 1.0f) * 255.0f),
        static_cast<uint8_t>(std::clamp(finalColorF[2], 0.0f, 1.0f) * 255.0f),
        255
    };
    return pix;
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
    const auto V = 1 - (projPos[1] * 0.5f + 0.5f);
    if (U < 0.0f || U > 1.0f || V < 0.0f || V > 1.0f)
        return 1.0f;

    constexpr float biasConst = 0.001f;
    constexpr float biasSlope = 0.01f;
    const auto N = normalize(normal);  // 法线
    const VecN<3> fragPos = {worldPos[0], worldPos[1], worldPos[2]}; // 片元世界坐标
    const auto L = normalize(mainLight->getPosi() - fragPos);  // 光照方向
    const float bias = biasConst + biasSlope * (1.0f - dot(N, L));

    const auto Z = projPos[2];  // 片元在光源空间的深度(NDC空间)
    // const float depth = ShadowMap.Sample(U, V);
    // const float shadow = (Z - bias > depth) ? 0.0f : 1.0f;
    // const float shadow = ShadowMap.SamplePCSS(Z, bias, U, V,0.005f, 3, 1, 15);
    const float shadow = ShadowMap.SamplePCF(Z, bias, U, V, 1);
    return shadow;
}