//
// Created by 冬榆 on 2025/12/27.
//

#ifndef UNTITLED_MAT_HPP
#define UNTITLED_MAT_HPP
#include "Vec.hpp"

// M行 N列,一列M行,一行N列. 按行优先存储
template<size_t M, size_t N>
struct MatMN {
    std::array<VecN<N>, M> data{};  // 存储M个N维行向量,data[i][j]访问i行j列

    MatMN(){data.fill(VecN<N>(0.0f));}
    explicit MatMN(const float a){data.fill(VecN<N>(a));}
    MatMN(const MatMN&) = default;
    MatMN& operator=(const MatMN&) = default;
    MatMN(MatMN&&) noexcept = default;
    template<typename... Args>
    explicit MatMN(Args&&... args)
        : data{std::forward<Args>(args)...}{
        static_assert(sizeof...(Args) == M, "Number of arguments must be exactly M.");
    }

    VecN<N>& operator[](size_t row) { return data[row]; }
    const VecN<N>& operator[](size_t row) const { return data[row]; }

    [[nodiscard]] const float* ptr() const { return &data[0][0]; }
    float* ptr() { return &data[0][0]; }

    MatMN& operator = (MatMN&& other) noexcept = default;

    MatMN<N, M> Transpose() const;  // 获取转置
};

using Mat4 = MatMN<4,4>;
using Mat3 = MatMN<3,3>;

// 矩阵乘法
template<size_t M, size_t K, size_t N>
MatMN<M, N> operator * (const MatMN<M, K> &lhs, const MatMN<K, N>& rhs);

// 运算符重载定义
template<size_t M, size_t N>
MatMN<M, N> operator + (const MatMN<M, N> &lhs, const MatMN<M, N> &rhs) {
    MatMN<M, N> result{};
    for (size_t i = 0; i < M; i++) {  // 先逐个访问行(i行号)
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> operator + (const MatMN<M, N> &lhs, float scalar) {
    MatMN<M, N> result{};
    for (size_t i = 0; i < M; i++) result[i] = scalar;
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> operator - (const MatMN<M, N> &lhs, const MatMN<M, N> &rhs) {
    MatMN<M, N> result{};
    for (size_t i = 0; i < M; i++)result[i] = lhs[i] - rhs[i];
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> operator - (const MatMN<M, N> &lhs, float scalar) {
    MatMN<M, N> result{};
    for (size_t i = 0; i < M; i++)result[i] = lhs[i] - scalar;
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> operator * (const MatMN<M, N> &lhs, float scalar) {
    MatMN<M, N> result{};
    for (size_t i = 0; i < M; i++) result[i] = lhs[i] * scalar;
    return result;
}

template<size_t M, size_t N>
VecN<M> operator * (const MatMN<M, N> &lhs, const VecN<N> &vector) {
    VecN<M> result;
    for (size_t i = 0; i < M; i++) result[i] = lhs[i] * vector;
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> operator / (const MatMN<M, N> &lhs, float scalar) {
    MatMN<M, N> result{};
    for (size_t i = 0; i < M; i++) result[i] = lhs[i] / scalar;
    return result;
}

template<size_t M, size_t N>
bool operator == (const MatMN<M, N> &lhs, const MatMN<M, N> &rhs) {
    for (size_t i = 0; i < M; i++) if (lhs[i] != rhs[i]) return false;
    return true;
}

template<size_t M, size_t N>
bool operator > (const MatMN<M, N> &lhs, float scalar) {
    for (size_t i = 0; i < M; i++) if (!(lhs[i] > scalar)) return false;
    return true;
}

// 优化写法是读取 lhs[i][k] 后，将其与 rhs 的第 k 行整行相乘累加
template<size_t M, size_t K, size_t N>
MatMN<M, N> operator * (const MatMN<M, K> &lhs, const MatMN<K, N>& rhs) {
    MatMN<M, N> result; // 默认初始化为0
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float r = lhs[i][k];
            for (size_t j = 0; j < N; ++j) {
                result[i][j] += r * rhs[k][j];
            }
        }
    }
    return result;
}

// template<size_t M, size_t N>
// MatMN<M, N>& MatMN<M, N>::operator=(MatMN&& other) noexcept {
    // if (this != &other) {
        // data = std::move(other.data);
    // }
    // return *this;
// }


template<size_t M, size_t N>
MatMN<N,M> MatMN<M,N>::Transpose() const {
    MatMN<N,M> result{};
    for(size_t i = 0; i < M; ++i) {
        for(size_t j = 0; j < N; ++j) {
            result[j][i] = data[i][j]; // 行列交换
        }
    }
    return result;
}


#endif //UNTITLED_MAT_HPP