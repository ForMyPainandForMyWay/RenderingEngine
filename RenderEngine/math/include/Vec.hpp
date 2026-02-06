#ifndef UNTITLED_VEXCOMPUTE_H
#define UNTITLED_VEXCOMPUTE_H

#include <array>
#include <cmath>
#include <algorithm>
#include <cstring>

template<size_t N>
struct alignas(N%4==0 ? 16 : 4) VecN {
    std::array<float, N> data;

    constexpr VecN() noexcept : data{} {}
    explicit constexpr VecN(float scalar) noexcept { data.fill(scalar); }
    explicit VecN(const float *arr) noexcept {
        std::memcpy(data.data(), arr, N * sizeof(float));
    }

    constexpr VecN(std::initializer_list<float> il) noexcept {
        const size_t count = std::min(N, il.size());
        auto it = il.begin();
        for(size_t i=0; i<count; ++i) data[i] = *it++;
    }

    static constexpr size_t getN() { return N; }

    // 访问器
    constexpr float& operator[] (size_t index) { return data[index]; }
    constexpr const float& operator[](size_t index) const { return data[index]; }

    // 获取裸指针 (用于 SIMD 加载)
    float* ptr() { return data.data(); }
    [[nodiscard]] const float* ptr() const { return data.data(); }

    // 长度平方，避免开根号
    [[nodiscard]] constexpr float lengthSq() const noexcept {
        float sum = 0.f;
        for (const float x : data) sum += x * x;
        return sum;
    }

    [[nodiscard]] float length() const noexcept {
        return std::sqrt(lengthSq());
    }

    // 归一化
    void normalize() noexcept {
        if (const float lenSq = lengthSq(); lenSq > 1e-8f) { // 避免除以 0
            float invLen = 1.0f / std::sqrt(lenSq);
            *this *= invLen;
        }
    }
};

using Vec3 = VecN<3>;
using Vec4 = VecN<4>;

// 运算符重载（移到结构体外并加上inline）
template<size_t N>
constexpr VecN<N> operator + (const VecN<N> &a, const VecN<N> &b) noexcept {
    VecN<N> res;
    for (size_t i = 0; i < N; ++i) res.data[i] = a.data[i] + b.data[i];
    return res;
}

template<size_t N>
constexpr VecN<N> operator - (const VecN<N> &a, const VecN<N> &b) noexcept {
    VecN<N> res;
    for (size_t i = 0; i < N; ++i) res.data[i] = a.data[i] - b.data[i];
    return res;
}

template<size_t N>
constexpr VecN<N> operator - (const VecN<N> &a, const float scalar) noexcept {
    VecN<N> res;
    for (size_t i = 0; i < N; ++i) res.data[i] = a.data[i] - scalar;
    return res;
}

template<size_t N>
constexpr VecN<N> operator * (const VecN<N> &a, float scalar) noexcept {
    VecN<N> res;
    for (size_t i = 0; i < N; ++i) res.data[i] = a.data[i] * scalar;
    return res;
}

template<size_t N>
constexpr VecN<N> operator / (const VecN<N> &a, const float scalar) noexcept {
    VecN<N> res;
    float inv = 1.0f / scalar;
    for (size_t i = 0; i < N; ++i) res.data[i] = a.data[i] * inv;
    return res;
}

template<size_t N>
constexpr float operator * (const VecN<N> &a, const VecN<N> &b) noexcept {
    float sum = 0.f;
    for (size_t i = 0; i < N; ++i) sum += a.data[i] * b.data[i];
    return sum;
}

template<size_t N>
constexpr void operator += (VecN<N> &a, const VecN<N> &b) noexcept {
    for (size_t i = 0; i < N; ++i) a.data[i] += b.data[i];
}

template<size_t N>
constexpr void operator *= (VecN<N> &a, float scalar) noexcept {
    for (size_t i = 0; i < N; ++i) a.data[i] *= scalar;
}

template<size_t N>
constexpr void operator /= (VecN<N> &a, const float scalar) noexcept {
    float inv = 1.0f / scalar;
    for (size_t i = 0; i < N; ++i) a.data[i] *= inv;
}

template<size_t N>
bool operator == (const VecN<N> &a, const VecN<N> &b) noexcept {
    for (size_t i = 0; i < N; ++i) {
        if (std::fabs(a.data[i] - b.data[i]) > 1e-5f) return false;
    }
    return true;
}

// 外部通用函数
// 归一化返回新向量
template<size_t N>
VecN<N> normalize(VecN<N> a) {
    a.normalize();
    return a;  // 触发 NRVO 优化，无额外拷贝
}

// 点乘
template<size_t N>
constexpr float dot(const VecN<N> &a, const VecN<N> &b) {
    return a * b;
}

// 叉乘
template<size_t N>
constexpr VecN<N> cross(const VecN<N> &a, const VecN<N> &b) {
    VecN<N> result{};
    if constexpr (N == 3) {
        result[0] = a[1] * b[2] - a[2] * b[1];
        result[1] = a[2] * b[0] - a[0] * b[2];
        result[2] = a[0] * b[1] - a[1] * b[0];
    } else if constexpr (N == 2) result[0] = a[0] * b[1] - a[1] * b[0];
    return result;
}

// 投影
template<size_t N>
VecN<N> project(const VecN<N> &a, const VecN<N> &b) {
    float bLenSq = b.lengthSq();
    if (bLenSq < 1e-8f) return VecN<N>{}; // 避免除零
    return b * ( (a * b) / bLenSq ); // 注意：这里不需要 sqrt
}

// Hadamard积 (逐分量乘法)
template<size_t N>
constexpr VecN<N> Hadamard(const VecN<N> &a, const VecN<N> &b) {
    VecN<N> result{};
    for (size_t i = 0; i < N; i++) result[i] = a[i] * b[i];
    return result;
}

#endif //UNTITLED_VEXCOMPUTE_H