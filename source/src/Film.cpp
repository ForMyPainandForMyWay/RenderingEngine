//
// Created by yyd on 2025/12/24.
//

#include "Film.h"

Film::Film(const size_t width, const std::size_t height) {
    this->width = width;
    this->height = height;
    // 通道组织方式为R:0 G:1 B:2
    this->image = new Pixel[width * height];
}

Film::~Film() {
    delete[] image;
}

// xy起始索引为0
Pixel Film::getPixel(const size_t x, const size_t y) const {
    if (x >= this->width || y >= this->height) {
        throw std::out_of_range("Film::getPixel out of range");
    }
    return this->image[y * this->width + x];
}

// RGBA模式的PAM格式存储
void Film::save(const std::string &filename) const {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) { perror("fopen: can not open file"); return; }
    fprintf(fp,
    "P7\n"
    "WIDTH %zu\n"
    "HEIGHT %zu\n"
    "DEPTH 4\n"
    "MAXVAL 255\n"
    "TUPLTYPE RGB_ALPHA\n"
    "ENDHDR\n",
    this->width, this->height
    );
    fwrite(this->image, sizeof(Pixel), width * height, fp);
    fclose(fp);
}

void Film::setPixel(const size_t x, const size_t y, const Pixel pixel) const {
    if (x >= this->width || y >= this->height) {
        throw std::out_of_range("Film::setPixel out of range");
    }
    this->image[y * this->width + x] = pixel;
}

void Film::setPixel(const size_t x, const size_t y,
                    const uint8_t r,
                    const uint8_t g,
                    const uint8_t b,
                    const uint8_t a) const {
    if (x >= this->width || y >= this->height) {
        throw std::out_of_range("Film::setPixel out of range");
    }
    this->image[y * this->width + x].r = r;
    this->image[y * this->width + x].g = g;
    this->image[y * this->width + x].b = b;
    this->image[y * this->width + x].a = a;
}

void Film::copyFromPtr(const unsigned char *data) const {
    std::memcpy(this->image, data, width * height * 4);
}
