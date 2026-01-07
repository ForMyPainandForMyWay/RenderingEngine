//
// Created by yyd on 2025/12/24.
//

#include "Film.h"

#include "F2P.h"

Film::Film(const size_t width, const std::size_t height) {
    this->width = width;
    this->height = height;
    // 通道组织方式为R:0 G:1 B:2
    // this->image.reserve(width * height);
    image.resize(width * height);
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

void Film::WritePixle(const F2P& f2p) {
    image[f2p.y * width + f2p.x] = f2p.color;
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
