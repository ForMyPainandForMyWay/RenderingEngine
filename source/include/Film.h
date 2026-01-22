//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_FILM_H
#define UNTITLED_FILM_H

#include <filesystem>

#include "Vec.hpp"
#include "VecPro.hpp"


struct Pixel;
struct F2P;

enum class Channel: uint8_t {
    R = 0,
    G = 1,
    B = 2,
};

struct FloatPixel {
    float r;
    float g;
    float b;
    float i;
    void operator+=(const FloatPixel& other);
    FloatPixel operator*(Vec3 K) const;
    void operator*=(Vec3 K);
    [[nodiscard]] Vec3 toFloat() const;
    [[nodiscard]] Pixel toPixel() const;
};

// 像素点,内存顺序布局
struct Pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;  // alpha or padding

    [[nodiscard]] Vec3 toFloat() const;
    [[nodiscard]] FloatPixel toFloatPixel() const;
};
static_assert(sizeof(Pixel) == 4);
static_assert(std::is_standard_layout_v<Pixel>);


// 胶片
struct Film {
    Film(size_t width, std::size_t height);

    void save(const std::string &filename) const;
    void copyFromPtr(const unsigned char *data);
    void clear();
    void fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a=255);
    void Trans2FloatPixel();

    uint32_t width, height;
    std::vector<Pixel> image;  // 以0为起始索引(为了适配帧缓冲使用指针)
    std::vector<FloatPixel> floatImg;  // 浮点化
};


#endif //UNTITLED_FILM_H