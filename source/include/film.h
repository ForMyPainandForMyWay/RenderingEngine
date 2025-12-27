//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_FILM_H
#define UNTITLED_FILM_H
#include <filesystem>


enum class Channel: uint8_t {
    R = 0,
    G = 1,
    B = 2,
};

// 像素点,内存顺序布局
struct Pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;  // alpha or padding
};
static_assert(sizeof(Pixel) == 4);
static_assert(std::is_standard_layout_v<Pixel>);


// 胶片
class Film {
public:
    Film(size_t width, std::size_t height);
    ~Film();

    void save(const std::string &filename) const;
    void setPixel(size_t x, size_t y, Pixel pixel) const;
    void setPixel(size_t x, size_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a=255) const;
    void copyFromPtr(const unsigned char *data) const;
    [[nodiscard]] size_t get_width() const{return this->width;}
    [[nodiscard]] size_t get_height() const{return this->height;}
    [[nodiscard]] Pixel getPixel(size_t x, size_t y) const;

private:
    size_t width, height;
    Pixel *image;  // 以0为起始索引(为了适配帧缓冲使用指针)
};


#endif //UNTITLED_FILM_H