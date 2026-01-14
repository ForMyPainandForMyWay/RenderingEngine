//
// Created by 冬榆 on 2026/1/9.
//

#include "GBuffer.h"

GBuffer::GBuffer(const size_t w, const size_t h) :
    width(w), height(h) {
    AlbedoBuffer.resize(w * h);
    MetallicBuffer.resize(w * h);
    RoughnessBuffer.resize(w * h);
    NormalBuffer.resize(w * h);
    WorldPosiBuffer.resize(w * h);
    DepthBuffer.resize(w * h);
    this->clear();
}

void GBuffer::WriteGBuffer(const GBufferData &data) {
    const auto locate = data.pixelCoord[0] + data.pixelCoord[1] * width;
    AlbedoBuffer[locate] = data.Albedo;
    MetallicBuffer[locate] = data.metallic;
    RoughnessBuffer[locate] = data.roughness;
    NormalBuffer[locate] = data.normal;
    WorldPosiBuffer[locate] = data.worldPosi;
    DepthBuffer[locate] = data.depth;
}

void GBuffer::WriteGBuffer(const std::vector<GBufferData> &dataVec) {
    for (const auto &data : dataVec) {
        const auto locate = data.pixelCoord[0] + data.pixelCoord[1] * width;
        AlbedoBuffer[locate] = data.Albedo;
        MetallicBuffer[locate] = data.metallic;
        RoughnessBuffer[locate] = data.roughness;
        NormalBuffer[locate] = data.normal;
        WorldPosiBuffer[locate] = data.worldPosi;
        DepthBuffer[locate] = data.depth;
    }
}

void GBuffer::clear() {
    std::ranges::fill(AlbedoBuffer, Pixel(0,0,0,0));
    std::ranges::fill(MetallicBuffer, VecN<3>{0,0,0});
    std::ranges::fill(RoughnessBuffer, VecN<3>{0,0,0});
    std::ranges::fill(NormalBuffer, VecN<3>{0,0,0});
    std::ranges::fill(WorldPosiBuffer, VecN<4>{0,0,0,0});
    std::ranges::fill(DepthBuffer, 1.0f);
}