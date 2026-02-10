//
// Created by 冬榆 on 2025/12/26.
//

#ifndef UNTITLED_PNGLOADER_H
#define UNTITLED_PNGLOADER_H

#include <string>
#include "Film.hpp"

std::unique_ptr<Film> loadImg(const std::string &path);


#endif //UNTITLED_PNGLOADER_H