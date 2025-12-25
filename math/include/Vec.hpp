//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_VEXCOMPUTE_H
#define UNTITLED_VEXCOMPUTE_H
#include <array>
#include <cmath>
#include <cstdint>
#include <iosfwd>
// struct Vec3;
// struct Vec2;
//
// struct Vec3 {
//     float x, y, z;
//
//     Vec3 operator + (const Vec3 &other) const;
//     Vec3 operator - (const Vec3 &other) const;
//     Vec3 operator * (float scalar) const;
//     Vec3 operator / (float scalar) const;
//
//     float operator * (const Vec3 &other) const;
//
//     bool operator == (const Vec3 &other) const;
//     bool operator > (float scalar) const;
// };
//
// struct Vec2 {
//     float x, y;
//
// };
//
//
template<size_t N>
struct VecN {
    std::array<float, N> data;

    VecN() {data.fill(0.0f);}

    explicit VecN(float scalar) {data.fill(scalar);}

    explicit VecN(float *arr) {
        std::copy(arr, arr + N, data.begin());
    }

    static float getN();


    float& operator[] (size_t index);
    const float& operator[](size_t index) const;
    VecN operator + (const VecN &other) const;
    VecN operator - (const VecN &other) const;
    VecN operator * (float scalar) const;
    float operator * (const VecN &other) const;
    VecN operator / (float scalar) const;
    bool operator == (const VecN &other) const;
    bool operator > (float scalar) const;

    template<size_t M>
    friend std::istream& operator>>(std::istream& is, VecN<M>& vec);

};

template<size_t N>
float VecN<N>::getN() {
    return N;
}

template<size_t N>
float& VecN<N>::operator[](size_t index) {
    return data[index];
}

template<size_t N>
const float& VecN<N>::operator[](size_t index) const{
    return data[index];
}

template<size_t N>
VecN<N> VecN<N>::operator+(const VecN &other) const {
    VecN result{};
    for (size_t i = 0; i < N; i++) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

template<size_t N>
VecN<N> VecN<N>::operator-(const VecN &other) const {
    VecN result{};
    for (size_t i = 0; i < N; i++) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

template<size_t N>
VecN<N> VecN<N>::operator*(float scalar) const {
    VecN result{};
    for (size_t i = 0; i < N; i++) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

template<size_t N>
VecN<N> VecN<N>::operator/(float scalar) const {
    VecN result{};
    for (size_t i = 0; i < N; i++) {
        result.data[i] = data[i] / scalar;
    }
    return result;
}

template<size_t N>
float VecN<N>::operator*(const VecN &other) const {
    float result{0.f};
    for (size_t i = 0; i < N; i++) {
        result += data[i] * other.data[i];
    }
    return result;
}

// 比较器，注意浮点数精度问题
template<size_t N>
bool VecN<N>::operator==(const VecN &other) const {
    for (size_t i = 0; i < N; ++i) {
        if (std::fabs(data[i] - other.data[i]) > 1e-6f) {
            return false;
        }
    }
    return true;
}

template<size_t N>
bool VecN<N>::operator>(float scalar) const {
    for (size_t i = 0; i < N; ++i) {
        if (data[i] <= scalar) return false;
    }
    return true;
}

template<size_t M>
std::istream& operator>>(std::istream& is, VecN<M>& vec) {
    for (size_t i = 0; i < M; ++i) {
        if (!(is >> vec.data[i])) break;
    }
    return is;
}
// // Vec3版本
// float getLength(const Vec3 &a);
// float dot(const Vec3 &a, const Vec3 &b);
// float cosAngle(const Vec3 &a, const Vec3 &b);
//
// bool sameDirection(const Vec3 &a, const Vec3 &b);
// bool inLeft(const Vec3 &a, const Vec3 &b);
//
// Vec3 project(const Vec3 &a, const Vec3 &b);
// Vec3 cross(const Vec3 &a, const Vec3 &b);
// Vec3 normalize(const Vec3 &a);


// 通用泛型
template<size_t N>
float getLength(const VecN<N> &a) {
    float result{};
    for (size_t i = 0; i < N; i++) {
        result += a.data[i] * a.data[i];
    }
    return std::sqrt(result);
}

template<size_t N>
float dot(const VecN<N> &a, const VecN<N> &b) {
    return a * b;
}

template<size_t N>
float cosAngle(const VecN<N> &a, const VecN<N> &b) {
    const float lenA = getLength(a);
    const float lenB = getLength(b);
    if (lenA == 0 || lenB == 0) return INFINITY;
    return a * b / (lenA * lenB);
}

template<size_t N>
bool sameWay(const VecN<N> &a, const VecN<N> &b) {
    return a * b > 0;
}

// 仅限二维向量比较
template<size_t N>
bool inLeft(const VecN<N> &a, const VecN<N> &b) {
    static_assert(N == 2, "inLeft() is only defined for 2D vectors");
    return a[0] * b[1] - a[1] * b[0] > 0;
}

template<size_t N>
bool inRight(const VecN<N> &a, const VecN<N> &b) {
    static_assert(N == 2, "inRight() is only defined for 2D vectors");
    return a[0] * b[1] - a[1] * b[0] < 0;
}

template<size_t N>
VecN<N> project(const VecN<N> &a, const VecN<N> &b) {
    return b  * ( a * b / (b * b) );
}

template<size_t N>
VecN<N> cross(const VecN<N> &a, const VecN<N> &b) {
    if constexpr (N == 2) {
        // 2D叉乘返回标量（伪标量）
        VecN<N> result{};
        result[0] = a[0] * b[1] - a[1] * b[0];
        return result;
    }
    else if constexpr (N == 3) {
        // 3D标准叉乘
        VecN<3> result;
        result[0] = a[1] * b[2] - a[2] * b[1];
        result[1] = a[2] * b[0] - a[0] * b[2];
        result[2] = a[0] * b[1] - a[1] * b[0];
        return result;
    }
    else {
        // 编译时错误或运行时错误
        static_assert(N == 2 || N == 3,
            "cross() is only defined for 2D and 3D vectors");
        return VecN<1>{};
    }
}

template<size_t N>
VecN<N> normalize(const VecN<N> &a) {
    return a / getLength(a);
}

#endif //UNTITLED_VEXCOMPUTE_H