//
// Created by 冬榆 on 2026/1/9.
//

#include "GBuffer.h"
#include "Shape.h"

GBuffer::GBuffer(const size_t w, const size_t h) : w(w), h(h) {
    Gdata.resize(w * h, GBufferData{VecN<3>{0.f, 0.f, 0.f}, VecN<4>{1.f, 1.f, 1.f, 1.f}});
}

void GBuffer::clear() {
    FragMap.clear();
    std::ranges::fill(Gdata, GBufferData{VecN<3>{0.f, 0.f, 0.f}, VecN<4>{1.f, 1.f, 1.f, 1.f}});
}
