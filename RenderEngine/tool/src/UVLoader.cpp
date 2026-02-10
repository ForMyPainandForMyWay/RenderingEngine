//
// Created by 冬榆 on 2025/12/26.
//

#define STB_IMAGE_IMPLEMENTATION

#include <iostream>

#include "UVLoader.hpp"
#include "stb_image.h"


// 加在PNG图像到uv中
std::unique_ptr<Film> loadPNG(const std::string &path) {
    int width=0, height=0, channel=4;

    if (!stbi_info(path.c_str(), &width, &height, &channel)) {
        std::cerr << "can not open texture png: " << path << std::endl;
    }

    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channel, 4);
    if (data == nullptr) std::cerr << "can not load png: "<< path << std::endl;

    // 拷贝数据
    if (data == nullptr) {
        width = height = 1024;
        return std::make_unique<Film>(width, height);
    }
    auto uvImg = std::make_unique<Film>(width, height);
    uvImg->copyFromPtr(data);
    stbi_image_free(data);
    return uvImg;
}