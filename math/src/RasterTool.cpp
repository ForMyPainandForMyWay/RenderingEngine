//
// Created by 冬榆 on 2026/1/7.
//

#include "RasterTool.hpp"
#include "Shape.h"


// 屏幕坐标三角形退化检测：三点面积是否为0，注意第一个的误差项是0.5,第二个误差项是1.0
void DegenerateClip(Triangle &tri) {
    float y0 = tri[0].clipPosi[1];
    float y1 = tri[1].clipPosi[1];
    float y2 = tri[2].clipPosi[1];

    // 简单退化
    if (std::max({y0, y1, y2}) - std::min({y0, y1, y2}) < 0.5f){
        tri.alive = false;
        return;
    }
    // 叉乘求面积,cross2D函数自动利用低二维计算叉乘
    if (fabs( crossInLow2D(tri[1].clipPosi-tri[0].clipPosi, tri[2].clipPosi-tri[0].clipPosi)) < 1.0f) {
        tri.alive = false;
    }
}

// 用于扫描线的三角形排序，不考虑退化情况（注意对应原点左上的屏幕坐标系）
void sortTriangle(Triangle &tri) {
    auto& v0 = tri[0];
    auto& v1 = tri[1];
    auto& v2 = tri[2];
    auto cmp = [](const V2F& a, const V2F& b) {
        if (fabs(a.clipPosi[1] - b.clipPosi[1]) > 1e-6) {
            return a.clipPosi[1] < b.clipPosi[1];}  // y 小在上
        return a.clipPosi[0] < b.clipPosi[0];       // y 相同，x 小在左
    };
    if (!cmp(v0, v1)) std::swap(v0, v1);
    if (!cmp(v1, v2)) std::swap(v1, v2);
    if (!cmp(v0, v1)) std::swap(v0, v1);
}

inline float EdgeFunc(const VecN<2>& a,
                      const VecN<2>& b,
                      const VecN<2>& p){
    return crossInLow2D(p-a, b-a);
}

inline bool IsTopLeft(const VecN<2>& a,
                      const VecN<2>& b){
    // 上边：y 相等，x 从左到右
    // 左边：y 从上到下
    return (std::abs((a[1]-b[1])) < 1e-5 && a[0] < b[0]) ||
           (a[1] > b[1]);
}

void Barycentric(Triangle& tri,
                         std::vector<Fragment>& result)
{
    if (!tri.alive) return;

    // 屏幕空间坐标
    const VecN<2> p0 = { tri[0].clipPosi[0], tri[0].clipPosi[1] };
    VecN<2> p1 = { tri[1].clipPosi[0], tri[1].clipPosi[1] };
    VecN<2> p2 = { tri[2].clipPosi[0], tri[2].clipPosi[1] };

    // AABB
    const int minX = static_cast<int>(std::floor(std::min({p0[0], p1[0], p2[0]})));
    const int maxX = static_cast<int>(std::ceil(std::max({p0[0], p1[0], p2[0]})));
    const int minY = static_cast<int>(std::floor(std::min({p0[1], p1[1], p2[1]})));
    const int maxY = static_cast<int>(std::ceil(std::max({p0[1], p1[1], p2[1]})));

    // 三角形面积
    float area = EdgeFunc(p0, p1, p2);
    if (std::abs(area) < 1e-6f)
        return;
    if (area < 1e-6f) {
        std::swap(p1, p2);
        std::swap(tri[1], tri[2]);
        area = -area;
    }

    const float invArea = 1.0f / area;

    // Top-Left 标记
    const bool tl01 = IsTopLeft(p0, p1);
    const bool tl12 = IsTopLeft(p1, p2);
    const bool tl20 = IsTopLeft(p2, p0);

    // 光栅化
    for (int y = minY; y < maxY; ++y) {
        for (int x = minX; x < maxX; ++x) {

            const VecN<2> p = {
                static_cast<float>(x) + 0.5f,
                static_cast<float>(y) + 0.5f
            };

            const float w0 = EdgeFunc(p1, p2, p);
            const float w1 = EdgeFunc(p2, p0, p);
            const float w2 = EdgeFunc(p0, p1, p);

            // Top-Left inside test
            if (!(
                (w0 > 1e-5 || (std::abs(w0) < 1e-5f && tl12)) &&
                (w1 > 1e-5 || (std::abs(w1) < 1e-5f && tl20)) &&
                (w2 > 1e-5 || (std::abs(w2) < 1e-5f && tl01))
            )) continue;

            // 重心坐标
            const float lambda0 = w0 * invArea;
            const float lambda1 = w1 * invArea;
            const float lambda2 = w2 * invArea;

            // 透视校正
            const float invW =
                lambda0 * tri[0].invW +
                lambda1 * tri[1].invW +
                lambda2 * tri[2].invW;

            const float w = 1.0f / invW;

            Fragment frag;
            frag.alive = true;
            frag.x = x;
            frag.y = y;

            frag.worldPosi =
                (tri[0].worldPosi * (lambda0 * tri[0].invW) +
                 tri[1].worldPosi * (lambda1 * tri[1].invW) +
                 tri[2].worldPosi * (lambda2 * tri[2].invW)) * w;

            frag.normal =
                (tri[0].normal * (lambda0 * tri[0].invW) +
                 tri[1].normal * (lambda1 * tri[1].invW) +
                 tri[2].normal * (lambda2 * tri[2].invW)) * w;

            frag.uv =
                (tri[0].uv * (lambda0 * tri[0].invW) +
                 tri[1].uv * (lambda1 * tri[1].invW) +
                 tri[2].uv * (lambda2 * tri[2].invW)) * w;

            frag.depth =
                (lambda0 * tri[0].clipPosi[2] * tri[0].invW +
                 lambda1 * tri[1].clipPosi[2] * tri[1].invW +
                 lambda2 * tri[2].clipPosi[2] * tri[2].invW) * w;

            // 切线空间插值
            frag.MainLightOri =
                (tri[0].MainLightOri * (lambda0 * tri[0].invW) +
                 tri[1].MainLightOri * (lambda1 * tri[1].invW) +
                 tri[2].MainLightOri * (lambda2 * tri[2].invW)) * w;
            frag.CameraOri =
                (tri[0].CameraOri * (lambda0 * tri[0].invW) +
                 tri[1].CameraOri * (lambda1 * tri[1].invW) +
                 tri[2].CameraOri * (lambda2 * tri[2].invW)) * w;
            // 归一化插值后的向量
            frag.MainLightOri = normalize(frag.MainLightOri);
            frag.CameraOri = normalize(frag.CameraOri);
            for (size_t i = 0; i < 3; ++i) {
                frag.PixLightOri[i] =
                    (tri[0].PixLightOri[i] * (lambda0 * tri[0].invW) +
                     tri[1].PixLightOri[i] *  (lambda1 * tri[1].invW) +
                     tri[2].PixLightOri[i] * (lambda2 * tri[2].invW)) * w;
                frag.PixLightOri[i] = normalize(frag.PixLightOri[i]);
            }
            frag.VexLightF =
                (tri[0].VexLightF * (lambda0 * tri[0].invW) +
                 tri[1].VexLightF * (lambda1 * tri[1].invW) +
                 tri[2].VexLightF * (lambda2 * tri[2].invW)) * w;
            result.push_back(frag);
        }
    }
}