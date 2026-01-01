//
// Created by 冬榆 on 2025/12/29.
//

#include "Engine.h"
#include "MathTool.hpp"
#include "RenderObjects.h"
#include "Uniform.h"


// 应用阶段，对实例应用变换
void Engine::Application() {
    while (!tfCommand.empty()) {
        const auto [objId, type, value] = tfCommand.front();
        tfCommand.pop();
        // 泛型 lambda：接受任意具有 updateP/Q/S 接口的对象
        auto applyCmd = [&](auto& obj) {
            switch (type) {
                case TfCmd::TRANSLATE:
                    obj.updateP(value);
                    break;
                case TfCmd::ROTATE:
                    obj.updateQ(Euler2Quaternion(value));
                    break;
                case TfCmd::SCALE:
                    obj.updateS(value);
                    break;
                default:
                    break;
            }
        };
        if (objId == 0) {
            applyCmd(camera);                     // camera 提供 updateP/Q/S
        } else {
            applyCmd(renderObjs.at(objId));   // renderObj 也提供相同接口
        }
    }
}

void Engine::DrawScene(const std::vector<uint8_t>& models) {
    auto PV = camera.ProjectionMat() * camera.ViewMat();
    globalU.setProjectView(PV);  // 更新全局Uniform
    for (const auto& model : models) {
        auto obj = renderObjs.at(model);
        auto uniform = Uniform(obj.updateMVP(PV),
                               obj.InverseTransposedMat());
        graphic.DrawModel(obj, uniform, 0);
    }
}