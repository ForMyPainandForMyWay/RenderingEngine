//
// Created by 冬榆 on 2026/1/9.
//

#include "GBuffer.hpp"
#include "Shape.hpp"

GBuffer::GBuffer(const size_t w, const size_t h) : w(w), h(h) {
    Gdata.resize(w * h, GBufferData{Vec3{0.f, 0.f, 0.f}, Vec4{1.f, 1.f, 1.f, 1.f}});
}

void GBuffer::clear() {
    FragMap.clear();
    std::ranges::fill(Gdata, GBufferData{Vec3{0.f, 0.f, 0.f}, Vec4{1.f, 1.f, 1.f, 1.f}});
}
