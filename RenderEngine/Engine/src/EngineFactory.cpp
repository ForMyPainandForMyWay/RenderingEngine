//
// Created by 冬榆 on 2026/2/16.
//

#include "Engine.hpp"
#include "IEngine.hpp"

// 工厂函数实现
extern "C" ENGINE_API IEngine *CreateEngine(const size_t w, const size_t h, const bool Gamma, const bool RT) {
    return new Engine(w, h, Gamma, RT);
}

extern "C" ENGINE_API void DestroyEngine(const IEngine* engine) {
    delete engine;
}