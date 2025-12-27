//
// Created by 冬榆 on 2025/12/27.
//

#ifndef UNTITLED_MAT_HPP
#define UNTITLED_MAT_HPP
#include "Vec.hpp"

// M行 N列,一列M行,一行N列. 按列优先存储
template<size_t M, size_t N>
struct MatMN {
    std::array<VecN<M>, N> data;  // 存储N个M维列向量,data[j][i]访问j列i行

    //MatMN() {data.fill(0.0f);}
    MatMN() {data.fill(VecN<M>(0.0f));}
    explicit MatMN(float scalar) {data.fill(VecN<M>(scalar));}
    // explicit MatMN(float *arr) {for (size_t j = 0; j < N; ++j) {data[j] = VecN<M>(arr + j * M);}}

    float get(size_t x, size_t y);
    VecN<M>& operator[](size_t col) { return data[col]; }
    const VecN<M>& operator[](size_t col) const { return data[col]; }

    MatMN operator + (const MatMN &other) const;
    MatMN operator + (float scalar) const;
    MatMN operator - (const MatMN &other) const;
    MatMN operator - (float scalar) const;
    MatMN operator * (float scalar) const;
    // float operator * (const MatMN &other) const;
    MatMN operator / (float scalar) const;
    bool operator == (const MatMN &other) const;
    bool operator > (float scalar) const;

    MatMN getTMat() const;  // 获取转置
};

// 矩阵乘法
template<size_t M, size_t K, size_t N>
MatMN<M, N> operator * (const MatMN<M, K> &lhs, const MatMN<K, N>& rhs);



// x是列号,y是行号.
template<size_t M, size_t N>
float MatMN<M, N>::get(size_t x, size_t y) {
    if (x >= M || y >= N) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[x][y]; // 列优先
}

template<size_t M, size_t N>
MatMN<M, N> MatMN<M, N>::operator + (const MatMN &other) const {
    MatMN result{};
    for (size_t i = 0; i < N; i++) {  // 先逐个访问列(i列号)
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
    for (size_t i = 0; i < N; i++) result[i] = scalar;
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> MatMN<M, N>::operator - (const MatMN &other) const {
    MatMN result{};
    for (size_t i = 0; i < N; i++)result[i] = (*this)[i] - other[i];
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> MatMN<M, N>::operator - (float scalar) const {
    MatMN result{};
    for (size_t i = 0; i < N; i++)result[i] = (*this)[i] - scalar;
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> MatMN<M, N>::operator * (float scalar) const {
    MatMN result{};
    for (size_t i = 0; i < N; i++) result[i] = (*this)[i] * scalar;
    return result;
}

template<size_t M, size_t K, size_t N>
MatMN<M, N> operator * (const MatMN<M, K> &lhs, const MatMN<K, N>& rhs) {
    MatMN<M, N> result{};
    for (size_t j = 0; j < N; ++j) {          // 遍历 rhs 的列
        for (size_t i = 0; i < M; ++i) {      // 遍历 lhs 的行
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {  // 累加点积
                sum += lhs[k][i] * rhs[j][k]; // 列优先
            }
            result[j][i] = sum; // 结果矩阵第j列第i行
        }
    }
    return result;
}

template<size_t M, size_t N>
MatMN<M, N> MatMN<M,N>::operator / (float scalar) const {
    MatMN result{};
    for (size_t i = 0; i < N; i++) result[i] = (*this)[i] / scalar;
    return result;
}

template<size_t M, size_t N>
bool MatMN<M, N>::operator == (const MatMN &other) const {
    for (size_t i = 0; i < N; i++) if ((*this)[i] != other[i]) return false;
    return true;
}

template<size_t M, size_t N>
bool MatMN<M, N>::operator > (float scalar) const {
    for (size_t i = 0; i < N; i++) if ((*this)[i] != scalar) return false;
    return true;
}

template<size_t M, size_t N>
MatMN<M, N> MatMN<M, N>::getTMat() const {
    MatMN result{};
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            result[i][j] = (*this)[j][i];
        }
    }
    return result;
}


#endif //UNTITLED_MAT_HPP