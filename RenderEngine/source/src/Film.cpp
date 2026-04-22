//
// Created by yyd on 2025/12/24.
//

#include "Film.hpp"


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
    image.resize(width * height, {255, 255, 255, 255});  // 默认白色背景
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

// BMP格式存储（24位RGB，不含alpha通道）
void Film::saveBMP(const std::string &filename) const {
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) { perror("fopen: can not open file"); return; }

    // BMP行需要4字节对齐
    const auto rowSize = ((width * 3 + 3) / 4) * 4;  // 每行字节数（对齐后）
    const auto padding = rowSize - width * 3;  // 每行填充字节数
    const auto imageSize = rowSize * height;
    const auto fileSize = 54 + imageSize;  // 文件头14字节 + 信息头40字节 + 图像数据

    // BMP文件头（14字节）
    unsigned char fileHeader[14] = {
        'B', 'M',                    // 签名
        0, 0, 0, 0,                  // 文件大小
        0, 0, 0, 0,                  // 保留
        54, 0, 0, 0                  // 像素数据偏移
    };
    fileHeader[2] = fileSize & 0xFF;
    fileHeader[3] = (fileSize >> 8) & 0xFF;
    fileHeader[4] = (fileSize >> 16) & 0xFF;
    fileHeader[5] = (fileSize >> 24) & 0xFF;

    // BMP信息头（40字节）
    unsigned char infoHeader[40] = {
        40, 0, 0, 0,                 // 信息头大小
        0, 0, 0, 0,                  // 宽度
        0, 0, 0, 0,                  // 高度
        1, 0,                        // 颜色平面数
        24, 0,                       // 每像素位数
        0, 0, 0, 0,                  // 压缩方式（无压缩）
        0, 0, 0, 0,                  // 图像大小
        0, 0, 0, 0,                  // 水平分辨率
        0, 0, 0, 0,                  // 垂直分辨率
        0, 0, 0, 0,                  // 颜色数
        0, 0, 0, 0                   // 重要颜色数
    };
    infoHeader[4] = width & 0xFF;
    infoHeader[5] = (width >> 8) & 0xFF;
    infoHeader[6] = (width >> 16) & 0xFF;
    infoHeader[7] = (width >> 24) & 0xFF;
    infoHeader[8] = height & 0xFF;
    infoHeader[9] = (height >> 8) & 0xFF;
    infoHeader[10] = (height >> 16) & 0xFF;
    infoHeader[11] = (height >> 24) & 0xFF;
    infoHeader[20] = imageSize & 0xFF;
    infoHeader[21] = (imageSize >> 8) & 0xFF;
    infoHeader[22] = (imageSize >> 16) & 0xFF;
    infoHeader[23] = (imageSize >> 24) & 0xFF;

    fwrite(fileHeader, 1, 14, fp);
    fwrite(infoHeader, 1, 40, fp);

    // BMP图像数据是倒序的（从下到上），且为BGR格式
    constexpr unsigned char paddingBytes[3] = {0, 0, 0};
    for (int y = static_cast<int>(height) - 1; y >= 0; --y) {
        for (size_t x = 0; x < width; ++x) {
            const Pixel& pixel = image[y * width + x];
            // BMP使用BGR顺序
            const unsigned char bgr[3] = {pixel.b, pixel.g, pixel.r};
            fwrite(bgr, 1, 3, fp);
        }
        // 写入填充字节
        if (padding > 0) {
            fwrite(paddingBytes, 1, padding, fp);
        }
    }

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