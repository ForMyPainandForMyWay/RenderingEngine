//
// Created by 冬榆 on 2026/1/9.
//

#ifndef RENDERINGENGINE_GBUFFER_H
#define RENDERINGENGINE_GBUFFER_H

#include "Film.h"
#include "Vec.hpp"


struct GBufferData {
    size_t pixelCoord[2]{};  // x, y
    float depth{};
    float AO{};  // 环境光遮蔽
    VecN<2> screenPosi;
    VecN<4> worldPosi;
    VecN<3> normal;
    VecN<2> uv;
    VecN<3> metallic{}; // 金属度
    VecN<3> roughness{};  // 粗糙度
    Pixel Albedo{};
};

class GBuffer {
public:
    GBuffer(size_t w, size_t h);
    void WriteGBuffer(const GBufferData &data);
    void WriteGBuffer(const std::vector<GBufferData> &dataVec);
    void clear();

    std::vector<Pixel> AlbedoBuffer;
    std::vector<VecN<3>> MetallicBuffer;
    std::vector<VecN<3>> RoughnessBuffer;
    std::vector<VecN<3>> NormalBuffer;
    std::vector<VecN<4>> WorldPosiBuffer;
    std::vector<float> DepthBuffer;


protected:
    size_t width;
    size_t height;
};


#endif //RENDERINGENGINE_GBUFFER_H