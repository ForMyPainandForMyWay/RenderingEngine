//
// Created by 冬榆 on 2025/12/29.
//

#include <cmath>
#include <future>
#include <ranges>

#include "RenderObjects.hpp"
#include "ClipTool.hpp"
#include "Engine.hpp"
#include "F2P.hpp"
#include "MathTool.hpp"
#include "Mesh.hpp"
#include "RasterTool.hpp"
#include "BlinnShader.hpp"


// 应用阶段，对实例应用变换
void Engine::Application() {
    while (!tfCommand.empty()) {
        auto [objId, typeId, type, value] = tfCommand.front();
        tfCommand.pop();
        // 泛型 lambda：接受任意具有 updateP/Q/S 接口的对象
        auto applyCmd = [&](auto& obj) {
            switch (type) {
                case TRANSLATE: obj.updateP(value);break;
                case ROTATE: obj.updateQ(Euler2Quaternion(value));break;  // 注意转弧度制
                case SCALE: obj.updateS(value);break;
                default: break;
            }
        };
        if (typeId == CameraID) applyCmd(camera);         // camera 提供 updateP/Q/S
        else if (typeId == MainLightID && mainLight!= nullptr) applyCmd(*mainLight);  // light 提供相同接口
        else if (typeId >= PixL1 && typeId <= PixL3 && PixLights[typeId-PixL1].alive) applyCmd(PixLights[objId-PixL1]);
        else if (typeId == RenderObject) applyCmd(renderObjs.at(objId));
        else if (typeId == VexLight) applyCmd(VexLights.at(objId));
        // else applyCmd(renderObjs.at(objId));  // renderObj 也提供相同接口
    }
}

// 几何着色阶段，对三角形进行几何处理
void Graphic::GeometryShading(Triangle &triangle, const std::shared_ptr<Material>& material, const Uniform &u, const int pass) const {
    if (!triangle.alive) return;
    if (material == nullptr) return;
    const auto shader = material->getShader(pass);
    if (shader == nullptr) return;

    const auto& PixLights = engine->PixLights;
    const auto& VexLights = engine->VexLights;
    const auto& mainLight = engine->mainLight;
    const auto& SdMap = engine->SdMap;
    const auto& envlight = engine->envLight;
    const auto& globalU = engine->globalU;
    shader->GeometryShader(
        triangle,
        material,
        PixLights,
        VexLights,
        mainLight,
        SdMap,
        envlight,
        globalU
    );
}

// ZTest组件
bool Graphic::ZTestPix(const size_t locate, const float depth, std::vector<float> &ZBuffer) {
    // 创建对 ZBuffer[locate] 的原子引用
    const std::atomic_ref zVal(ZBuffer[locate]);
    float oldZ = zVal.load(std::memory_order_relaxed);
    while (true) {
        if (oldZ < depth) return false;
        if (zVal.compare_exchange_weak(oldZ, depth, std::memory_order_relaxed)) {
            return true;
        }
    }
}

bool Graphic::Ztest(Fragment &frag, std::vector<float> &ZBuffer) const {
    if (!frag.alive) return false;
    const size_t locate = frag.x + frag.y * engine->width;
    if (ZTestPix(locate, frag.depth, ZBuffer)) {
        frag.keep();
        return true;
    }
    frag.drop();
    return false;
}