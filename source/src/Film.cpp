//
// Created by yyd on 2025/12/24.
//

#include "Film.h"

#include <complex>

#include "F2P.h"

void Pixel::operator+=(const Pixel& other) {
    this->r = std::min(255, static_cast<int>(this->r) + static_cast<int>(other.r));
    this->g = std::min(255, static_cast<int>(this->g) + static_cast<int>(other.g));
    this->b = std::min(255, static_cast<int>(this->b) + static_cast<int>(other.b));
    this->a = std::min(255, static_cast<int>(this->a) + static_cast<int>(other.a));
}

Pixel Pixel::operator*(VecN<3> K) const {
    Pixel result{};
    result.a = 255;
    result.r = static_cast<uint8_t>(std::clamp((static_cast<float>(r) * K[0]), 0.0f, 255.0f));
    result.g = static_cast<uint8_t>(std::clamp((static_cast<float>(g) * K[1]), 0.0f, 255.0f));
    result.b = static_cast<uint8_t>(std::clamp((static_cast<float>(b) * K[2]), 0.0f, 255.0f));
    return result;
}

void Pixel::operator*=(VecN<3> K) {
    this->r = static_cast<uint8_t>(std::clamp((static_cast<float>(r) * K[0]), 0.0f, 255.0f));
    this->g = static_cast<uint8_t>(std::clamp((static_cast<float>(g) * K[1]), 0.0f, 255.0f));
    this->b = static_cast<uint8_t>(std::clamp((static_cast<float>(b) * K[2]), 0.0f, 255.0f));
}

VecN<3> Pixel::toFloat() const {
    VecN<3> result{};
    result[0] = static_cast<float>(r) / 255.0f;
    result[1] = static_cast<float>(g) / 255.0f;
    result[2] = static_cast<float>(b) / 255.0f;
    return result;
}

Film::Film(const size_t width, const std::size_t height) {
    this->width = width;
    this->height = height;
    // 通道组织方式为R:0 G:1 B:2
    // this->image.reserve(width * height);
    image.resize(width * height);
    this->fill(255, 255, 255, 255);  // 默认白色背景
}

const Pixel& Film::getPixel(const size_t i) const {
     return this->image[i];
}

Pixel& Film::operator[](const size_t i) {
    return image[i];
}

const Pixel& Film::operator[](const size_t i) const {
    return image[i];
}

void Film::clear() {
    std::ranges::fill(image, Pixel(0, 0, 0, 255));
}

void Film::fill(const uint8_t r, const uint8_t g, const uint8_t b, const uint8_t a) {
    std::ranges::fill(image, Pixel(r, g, b, a));
}

void Film::WritePixle(const F2P& f2p) {
    image[f2p.y * width + f2p.x] = f2p.Albedo;
}

// RGBA模式的PAM格式存储
void Film::save(const std::string &filename) const {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) { perror("fopen: can not open file"); return; }
    fprintf(fp,
    "P7\n"
    "WIDTH %u\n"
    "HEIGHT %u\n"
    "DEPTH 4\n"
    "MAXVAL 255\n"
    "TUPLTYPE RGB_ALPHA\n"
    "ENDHDR\n",
    this->width, this->height
    );
    fwrite(image.data(), sizeof(Pixel), width * height, fp);
    fclose(fp);
}

void Film::copyFromPtr(const unsigned char *data) {
    std::memcpy(image.data(), data, width * height * sizeof(Pixel));;
}
