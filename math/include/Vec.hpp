//
// Created by yyd on 2025/12/24.
//

#ifndef UNTITLED_VEXCOMPUTE_H
#define UNTITLED_VEXCOMPUTE_H
#include <array>
#include <cmath>


// 非齐次坐标,运行时升纬度
template<size_t N>
struct VecN {
    std::array<float, N> data;

    explicit VecN(float scalar) {data.fill(scalar);}
    explicit VecN(float *arr) {std::copy(arr, arr+N, data.begin());}
    VecN(std::initializer_list<float> il) {std::copy(il.begin(), il.end(), data.begin());}
    VecN() : data{} {}

    static float getN();

    float& operator[] (size_t index);
    const float& operator[](size_t index) const;
    VecN operator + (const VecN &other) const;
    VecN operator + (float scalar) const;
    VecN operator - (const VecN &other) const;
    VecN operator - (float scalar) const;
    VecN operator * (float scalar) const;
    float operator * (const VecN &other) const;
    VecN operator / (float scalar) const;
    void operator += (const VecN &other);
    void operator *= (const VecN &other);
    void operator *= (const float scalar);
    void operator /= (float w);
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
VecN<N> VecN<N>::operator+(const float scalar) const {
    VecN result{};
    for (size_t i = 0; i < N; i++) {
        result.data[i] = data[i] + scalar;
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
VecN<N> VecN<N>::operator-(float scalar) const {
    VecN result{};
    for (size_t i = 0; i < N; i++) {
        result.data[i] = data[i] - scalar;
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
void VecN<N>::operator += (const VecN &other) {
    for (size_t i = 0; i < N; i++) {
        this->data[i] += other[i];
    }
}

template<size_t N>
void VecN<N>::operator *= (const VecN &other) {
    for (size_t i = 0; i < N; i++) {
        this->data[i] *= other[i];
    }
}

template<size_t N>
void VecN<N>::operator *= (const float scalar) {
    for (size_t i = 0; i < N; i++) {
        this->data[i] += scalar;
    }
}

template<size_t N>
void VecN<N>::operator /= (const float w) {
    for (size_t i = 0; i < N; i++) {
        this->data[i] /= w;
    }
}

template<size_t N>
float VecN<N>::operator*(const VecN &other) const {
    float result{0.f};
    for (size_t i = 0; i < N; i++) {
        result += data[i] * other.data[i];
    }
    return result;
}

// 比较器,注意浮点数精度问题
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


// 通用泛型
// 计算模长
template<size_t N>
float getLength(const VecN<N> &a) {
    float result{};
    for (size_t i = 0; i < N; i++) {
        result += a.data[i] * a.data[i];
    }
    return std::sqrt(result);
}

// 点乘
template<size_t N>
float dot(const VecN<N> &a, const VecN<N> &b) {
    return a * b;
}

// 计算夹角余弦值
template<size_t N>
float cosAngle(const VecN<N> &a, const VecN<N> &b) {
    const float lenA = getLength(a);
    const float lenB = getLength(b);
    if (lenA == 0 || lenB == 0) return INFINITY;
    return a * b / (lenA * lenB);
}

// 判断是否同向(非严格)
template<size_t N>
bool sameWay(const VecN<N> &a, const VecN<N> &b) {
    return a * b > 0;
}

// 是否在左侧 仅限二维向量比较
template<size_t N>
bool inLeft(const VecN<N> &a, const VecN<N> &b) {
    static_assert(N == 2, "inLeft() is only defined for 2D vectors");
    return a[0] * b[1] - a[1] * b[0] > 0;
}

// 是否在右侧 仅限二维
template<size_t N>
bool inRight(const VecN<N> &a, const VecN<N> &b) {
    static_assert(N == 2, "inRight() is only defined for 2D vectors");
    return a[0] * b[1] - a[1] * b[0] < 0;
}

// a 在 b 方向上的投影
template<size_t N>
VecN<N> project(const VecN<N> &a, const VecN<N> &b) {
    return b  * ( a * b / (b * b) );
}

// a x b 仅限二三维
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
        VecN<3> result{};
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

// 任意维度向量的低维坐标的叉积
template<size_t N>
float cross2D(const VecN<N> &a, const VecN<N> &b) {
    return a[0] * b[1] - a[1] * b[0];
}

// 标准化向量,返回新向量
template<size_t N>
VecN<N> normalize(const VecN<N> &a) {
    return a / getLength(a);
}

#endif //UNTITLED_VEXCOMPUTE_H