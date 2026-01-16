//
// Created by 冬榆 on 2025/12/27.
//

#ifndef UNTITLED_MAT_HPP
#define UNTITLED_MAT_HPP
#include "Vec.hpp"

// M行 N列,一列M行,一行N列. 按行优先存储
template<size_t M, size_t N>
struct MatMN {
    std::array<VecN<N>, M> data;  // 存储M个N维行向量,data[i][j]访问i行j列

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

    float get(size_t row, size_t col);
    VecN<N>& operator[](size_t row) { return data[row]; }
    const VecN<N>& operator[](size_t row) const { return data[row]; }

    MatMN operator + (const MatMN &other) const;
    MatMN operator + (float scalar) const;
    MatMN operator - (const MatMN &other) const;
    MatMN operator - (float scalar) const;
    MatMN operator * (float scalar) const;
    VecN<M> operator * (VecN<N>) const;
    MatMN operator / (float scalar) const;
    bool operator == (const MatMN &other) const;
    bool operator > (float scalar) const;
    MatMN& operator = (MatMN&& other) noexcept;

    MatMN<N, M> Transpose() const;  // 获取转置
    void show() {
        for (auto i=0; i < M; ++i) {
            for (auto j=0; j<N; ++j) {
                printf("%d ", data[i][j]);
            }
            printf("\n");
        }
    }
};

// 矩阵乘法
template<size_t M, size_t K, size_t N>
MatMN<M, N> operator * (const MatMN<M, K> &lhs, const MatMN<K, N>& rhs);



// x是行号,y是列号.
template<size_t M, size_t N>
float MatMN<M, N>::get(size_t row, size_t col) {
    if (row >= M || col >= N) throw std::out_of_range("Index out of bounds");
    return data[row][col]; // 行优先
}

template<size_t M, size_t N>
MatMN<M, N> MatMN<M, N>::operator + (const MatMN &other) const {
    MatMN result{};
    for (size_t i = 0; i < M; i++) {  // 先逐个访问行(i行号)
        result[i] = (*this)[i] + other[i];
        // for (size_t j = 0; j < M; j++) {  // 每一列内再逐行访问(j行号)
        // result[i][j] = (*this)[i][j] + other[i][j];
        // }
    }
    return result;
}

template<size_t M, size_t N>
MatMN<M,N> MatMN<M, N>::operator + (float scalar) const{
    MatMN result{};
    for (size_t i = 0; i < M; i++) result[i] = scalar;
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> MatMN<M, N>::operator - (const MatMN &other) const {
    MatMN result{};
    for (size_t i = 0; i < M; i++)result[i] = (*this)[i] - other[i];
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> MatMN<M, N>::operator - (float scalar) const {
    MatMN result{};
    for (size_t i = 0; i < M; i++)result[i] = (*this)[i] - scalar;
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> MatMN<M, N>::operator * (float scalar) const {
    MatMN result{};
    for (size_t i = 0; i < M; i++) result[i] = (*this)[i] * scalar;
    return result;
}

// [M,N]矩阵乘[N]向量
template<size_t M, size_t N>
VecN<M> MatMN<M, N>::operator*(VecN<N> vector) const {
    VecN<M> result;
    for (size_t i = 0; i < M; i++) result[i] = (*this)[i] * vector;
    return result;
}

template<size_t M, size_t K, size_t N>
MatMN<M,N> operator*(const MatMN<M,K> &lhs, const MatMN<K,N> &rhs) {
    MatMN<M,N> result{};
    for(size_t i = 0; i < M; ++i) {      // 遍历 lhs 行 / result 行
        for(size_t j = 0; j < N; ++j) {  // 遍历 rhs 列 / result 列
            float sum = 0.0f;
            for(size_t k = 0; k < K; ++k) {
                // 按行访问 lhs[i], rhs 列访问 rhs[k][j]
                sum += lhs[i][k] * rhs[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}


template<size_t M, size_t N>
MatMN<M, N> MatMN<M,N>::operator / (float scalar) const {
    MatMN result{};
    for (size_t i = 0; i < M; i++) result[i] = (*this)[i] / scalar;
    return result;
}

template<size_t M, size_t N>
bool MatMN<M, N>::operator == (const MatMN &other) const {
    for (size_t i = 0; i < M; i++) if ((*this)[i] != other[i]) return false;
    return true;
}

template<size_t M, size_t N>
bool MatMN<M, N>::operator > (float scalar) const {
    for (size_t i = 0; i < M; i++) if (!((*this)[i] > scalar)) return false;
    return true;
}

template<size_t M, size_t N>
MatMN<M, N>& MatMN<M, N>::operator=(MatMN&& other) noexcept {
    if (this != &other) {
        data = std::move(other.data);
    }
    return *this;
}


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