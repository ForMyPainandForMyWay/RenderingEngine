//
// Created by yyd on 2025/12/24.
//

#include "Film.hpp"

#include "VecPro.hpp"


void FloatPixel::operator+=(const FloatPixel& other) {
    r += other.r;
    g += other.g;
    b += other.b;
}

FloatPixel FloatPixel::operator*(Vec3 K) const {
    FloatPixel result{};
    result.r = r * K[0];
    result.g = g * K[1];
    result.b = b * K[2];
    return result;
}

void FloatPixel::operator*=(Vec3 K) {
    r *= K[0];
    g *= K[1];
    b *= K[2];
}

Vec3 FloatPixel::toFloat() const {
    return {r, g, b};
}

Pixel FloatPixel::toPixel() const {
    const auto r_ = static_cast<uint8_t>(std::clamp(r, 0.f, 1.f) * 255.0f);
    const auto g_ = static_cast<uint8_t>(std::clamp(g, 0.f, 1.f) * 255.0f);
    const auto b_ = static_cast<uint8_t>(std::clamp(b, 0.f, 1.f) * 255.0f);
    return {r_, g_, b_ , 255};
}

Vec3 Pixel::toFloat() const {
    Vec3 result{};
    result[0] = static_cast<float>(r) / 255.0f;
    result[1] = static_cast<float>(g) / 255.0f;
    result[2] = static_cast<float>(b) / 255.0f;
    return result;
}

FloatPixel Pixel::toFloatPixel() const {
    FloatPixel result{};
    result.r = static_cast<float>(r) / 255.0f;
    result.g = static_cast<float>(g) / 255.0f;
    result.b = static_cast<float>(b) / 255.0f;
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

void Film::clear() {
    std::ranges::fill(image, Pixel(0, 0, 0, 255));
}

void Film::fill(const uint8_t r, const uint8_t g, const uint8_t b, const uint8_t a) {
    std::ranges::fill(image, Pixel(r, g, b, a));
}

// RGBA模式的PAM格式存储
void Film::save(const std::string &filename) const {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) { perror("fopen: can not open file"); return; }
    fprintf(fp,
    "P7\n"
    "WIDTH %lu\n"
    "HEIGHT %lu\n"
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
    std::memcpy(image.data(), data, width * height * sizeof(Pixel));
}

// 将8位像素转位浮点数，填充到floatImg，清空image数据
void Film::Trans2FloatPixel() {
    this->floatImg.reserve(this->image.size());
    for (auto& pix : image) {
        this->floatImg.emplace_back(pix.toFloatPixel());
    }
    this->image.clear();
}