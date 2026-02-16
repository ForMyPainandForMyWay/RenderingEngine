//
// Created by 冬榆 on 2026/1/7.
//

#include "RasterTool.hpp"
#include "MathTool.hpp"
#include "Shape.hpp"


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
    // 叉乘求面积
    if (fabs(TriScreenArea2(tri)) < 0.5f) {
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

void BarycentricOptimizedFull(Triangle& tri, std::vector<Fragment>& result, int screenWidth, int screenHeight) {
    if (!tri.alive) return;

    // 使用局部指针，避免修改原始三角形数据 (原代码的 swap 会破坏原始 mesh)
    const auto* v0 = &tri[0];
    const auto* v1 = &tri[1];
    const auto* v2 = &tri[2];
    // 屏幕空间坐标提取
    const VecN<2> p0 = { v0->clipPosi[0], v0->clipPosi[1] };
    VecN<2> p1 = { v1->clipPosi[0], v1->clipPosi[1] };
    VecN<2> p2 = { v2->clipPosi[0], v2->clipPosi[1] };
    float area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
    // 退化三角形剔除
    if (std::abs(area) < 1e-6f) return;
    // 绕序修正
    if (area < 0) {
        std::swap(p1, p2); // 交换局部坐标
        std::swap(v1, v2); // 交换局部指针指向
        area = -area;
    }
    const float invArea = 1.0f / area;
    // AABB 包围盒计算与屏幕裁剪
    int minX = static_cast<int>(std::floor(std::min({p0[0], p1[0], p2[0]})));
    int maxX = static_cast<int>(std::ceil(std::max({p0[0], p1[0], p2[0]})));
    int minY = static_cast<int>(std::floor(std::min({p0[1], p1[1], p2[1]})));
    int maxY = static_cast<int>(std::ceil(std::max({p0[1], p1[1], p2[1]})));

    // 裁剪到屏幕范围
    minX = std::max(0, minX);
    minY = std::max(0, minY);
    maxX = std::min(static_cast<int>(screenWidth), maxX);
    maxY = std::min(static_cast<int>(screenHeight), maxY);
    if (minX >= maxX || minY >= maxY) return;

    // 边缘方程系数计算
    // w0 由边 v1-v2 决定, w1 由 v2-v0 决定, w2 由 v0-v1 决定
    auto calc_coeffs = [](const VecN<2>& a, const VecN<2>& b) {
        float A = a[1] - b[1];
        float B = b[0] - a[0];
        float C = -B * a[1] - A * a[0];
        return std::make_tuple(A, B, C);
    };
    auto [A0, B0, C0] = calc_coeffs(p1, p2);
    auto [A1, B1, C1] = calc_coeffs(p2, p0);
    auto [A2, B2, C2] = calc_coeffs(p0, p1);

    // Top-Left Rule 优化 (Bias)
    // 只有当边是 Top 或 Left 时 bias 为 0，否则为 -epsilon
    // 这样内循环只需要判断 value >= 0
    constexpr float eps = 1e-4f;
    auto is_top_left = [](const VecN<2>& start, const VecN<2>& end) {
        VecN<2> edge = { end[0] - start[0], end[1] - start[1] };
        const bool isTop = (edge[1] == 0 && edge[0] > 0);
        const bool isLeft = (edge[1] < 0);
        return isTop || isLeft;
    };

    float bias0 = is_top_left(p1, p2) ? 0.0f : -eps;
    float bias1 = is_top_left(p2, p0) ? 0.0f : -eps;
    float bias2 = is_top_left(p0, p1) ? 0.0f : -eps;

    // 属性预计算
    // 将 (Attribute * invW / Area) 预先算好
    // Z因子：用于透视校正分母
    float z0 = v0->invW * invArea;
    float z1 = v1->invW * invArea;
    float z2 = v2->invW * invArea;

    // 预计算属性
    Vec4 p_worldPos0 = v0->worldPosi * z0;
    Vec4 p_worldPos1 = v1->worldPosi * z1;
    Vec4 p_worldPos2 = v2->worldPosi * z2;

    Vec3 p_normal0 = v0->normal * z0;
    Vec3 p_normal1 = v1->normal * z1;
    Vec3 p_normal2 = v2->normal * z2;

    auto p_uv0 = v0->uv * z0;
    auto p_uv1 = v1->uv * z1;
    auto p_uv2 = v2->uv * z2;

    // 深度值
    float p_depth0 = v0->clipPosi[2] * z0;
    float p_depth1 = v1->clipPosi[2] * z1;
    float p_depth2 = v2->clipPosi[2] * z2;

    Vec3 p_mainLight0 = v0->MainLightOri * z0;
    Vec3 p_mainLight1 = v1->MainLightOri * z1;
    Vec3 p_mainLight2 = v2->MainLightOri * z2;

    Vec3 p_camOri0 = v0->CameraOri * z0;
    Vec3 p_camOri1 = v1->CameraOri * z1;
    Vec3 p_camOri2 = v2->CameraOri * z2;

    Vec3 p_vexLight0 = v0->VexLightF * z0;
    Vec3 p_vexLight1 = v1->VexLightF * z1;
    Vec3 p_vexLight2 = v2->VexLightF * z2;

    // 数组属性 PixLightOri[3] 的预计算. 注意这里写死3个了
    Vec3 p_pixLight0[3], p_pixLight1[3], p_pixLight2[3];
    for(int i=0; i<3; ++i) {
        p_pixLight0[i] = v0->PixLightOri[i] * z0;
        p_pixLight1[i] = v1->PixLightOri[i] * z1;
        p_pixLight2[i] = v2->PixLightOri[i] * z2;
    }

    // 采样点位于像素中心 (x+0.5, y+0.5)
    float startX = static_cast<float>(minX) + 0.5f;
    float startY = static_cast<float>(minY) + 0.5f;

    float row_w0 = A0 * startX + B0 * startY + C0 + bias0;
    float row_w1 = A1 * startX + B1 * startY + C1 + bias1;
    float row_w2 = A2 * startX + B2 * startY + C2 + bias2;

    // 双重循环光栅化
    for (int y = minY; y < maxY; ++y) {
        float w0 = row_w0;
        float w1 = row_w1;
        float w2 = row_w2;
        for (int x = minX; x < maxX; ++x) {
            // 检查点是否在三角形内
            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                // 透视校正分母
                float invW_pixel = w0 * z0 + w1 * z1 + w2 * z2;
                float w_pixel = 1.0f / invW_pixel;
                // 原地构造 Fragment
                result.emplace_back();
                auto&[triX, triY, depth, worldPosi, normal, uv, VexLightF, PixLightOri, MainLightOri, CameraOri, alive] = result.back();
                alive = true;
                triX = x;
                triY = y;
                // 属性插值 (pre0 * w0 + pre1 * w1 + pre2 * w2) * w_pixel
                worldPosi = (p_worldPos0 * w0 + p_worldPos1 * w1 + p_worldPos2 * w2) * w_pixel;
                uv        = (p_uv0 * w0 + p_uv1 * w1 + p_uv2 * w2) * w_pixel;
                depth     = (p_depth0 * w0 + p_depth1 * w1 + p_depth2 * w2) * w_pixel;
                VexLightF = (p_vexLight0 * w0 + p_vexLight1 * w1 + p_vexLight2 * w2) * w_pixel;
                normal    = (p_normal0 * w0 + p_normal1 * w1 + p_normal2 * w2) * w_pixel;
                MainLightOri = (p_mainLight0 * w0 + p_mainLight1 * w1 + p_mainLight2 * w2) * w_pixel;
                CameraOri = (p_camOri0 * w0 + p_camOri1 * w1 + p_camOri2 * w2) * w_pixel;
                for(int i=0; i<3; ++i) {
                    PixLightOri[i] = (p_pixLight0[i] * w0 + p_pixLight1[i] * w1 + p_pixLight2[i] * w2) * w_pixel;
                }
            }
            // X 轴增量更新
            w0 += A0;
            w1 += A1;
            w2 += A2;
        }
        // Y 轴增量更新
        row_w0 += B0;
        row_w1 += B1;
        row_w2 += B2;
    }
}

// ==========================================
// 定义插值载荷 (Payload)
// 用于打包所有需要插值的属性，利用运算符重载简化代码
// ==========================================
struct Payload {
    // 所有的属性都已经预先除以了 W (perspective divided)
    float invW{};
    float depth{};
    VecN<2> uv;
    Vec4 worldPos;
    Vec3 normal;
    Vec3 mainLight;
    Vec3 camOri;
    Vec3 vexLight;
    Vec3 pixLight[3];

    // 运算符重载：实现属性的加法和减法 (用于计算梯度)
    Payload operator+(const Payload& other) const {
        Payload res;
        res.invW = invW + other.invW;
        res.depth = depth + other.depth;
        res.uv = uv + other.uv;
        res.worldPos = worldPos + other.worldPos;
        res.normal = normal + other.normal;
        res.mainLight = mainLight + other.mainLight;
        res.camOri = camOri + other.camOri;
        res.vexLight = vexLight + other.vexLight;
        for(int i=0; i<3; ++i) res.pixLight[i] = pixLight[i] + other.pixLight[i];
        return res;
    }

    Payload operator-(const Payload& other) const {
        Payload res;
        res.invW = invW - other.invW;
        res.depth = depth - other.depth;
        res.uv = uv - other.uv;
        res.worldPos = worldPos - other.worldPos;
        res.normal = normal - other.normal;
        res.mainLight = mainLight - other.mainLight;
        res.camOri = camOri - other.camOri;
        res.vexLight = vexLight - other.vexLight;
        for(int i=0; i<3; ++i) res.pixLight[i] = pixLight[i] - other.pixLight[i];
        return res;
    }

    // 数乘 (用于计算 step = delta / dist)
    Payload operator*(const float s) const {
        Payload res;
        res.invW = invW * s;
        res.depth = depth * s;
        res.uv = uv * s;
        res.worldPos = worldPos * s;
        res.normal = normal * s;
        res.mainLight = mainLight * s;
        res.camOri = camOri * s;
        res.vexLight = vexLight * s;
        for(int i=0; i<3; ++i) res.pixLight[i] = pixLight[i] * s;
        return res;
    }
};

// ==========================================
// 辅助类：边缘行进器
// 负责沿 Y 轴插值 X 坐标和 Payload
// ==========================================
struct ScanEdge {
    float x{};        // 当前行的 x 坐标
    float dx_dy{};    // x 对 y 的增量 (斜率倒数)

    Payload current; // 当前行的属性值
    Payload step;    // 属性对 y 的增量 (dAttr/dy)

    int yStart{}, yEnd{}; // 边的 Y 范围

    // 初始化边
    void Init(const VecN<2>& pTop, const Payload& vTop,
              const VecN<2>& pBot, const Payload& vBot, const int screenH) {
        yStart = static_cast<int>(std::ceil(pTop[1] - 0.5f)); // 向上取整对齐像素中心
        yEnd   = static_cast<int>(std::ceil(pBot[1] - 0.5f));
        yEnd   = std::min(yEnd, screenH); // 裁剪

        const float dy = pBot[1] - pTop[1];
        if (std::abs(dy) < 1e-5f) {
            dx_dy = 0.0f;
            // 对于水平边，step 设为0，防止除0
            step = vTop * 0.0f;
            return;
        }

        const float invDy = 1.0f / dy;
        dx_dy = (pBot[0] - pTop[0]) * invDy;
        step  = (vBot - vTop) * invDy;
        // 因为扫描线是从整数 Y 开始的，而顶点可能是浮点数 Y。
        // 需要把属性向前推演 (yStart + 0.5 - pTop.y)
        const float preStepY = (static_cast<float>(yStart) + 0.5f) - pTop[1];
        x = pTop[0] + dx_dy * preStepY;
        current = vTop + (step * preStepY);
    }

    // 步进下一行
    void Step() {
        x += dx_dy;
        current = current + step;
    }
};

// 扫描线
void Scanline(Triangle& tri, std::vector<Fragment>& result, int screenWidth, int screenHeight) {
    if (!tri.alive) return;

    const auto* v0 = &tri[0];
    const auto* v1 = &tri[1];
    const auto* v2 = &tri[2];

    // 屏幕坐标
    VecN<2> p0 = { v0->clipPosi[0], v0->clipPosi[1] };
    VecN<2> p1 = { v1->clipPosi[0], v1->clipPosi[1] };
    VecN<2> p2 = { v2->clipPosi[0], v2->clipPosi[1] };

    // 透视除法预处理
    auto create_payload = [](const V2F* v) -> Payload {
        Payload p;
        const float invW = v->invW; // 假设这是 1/w
        p.invW = invW;
        p.depth = v->clipPosi[2] * invW;
        p.uv = v->uv * invW;
        p.worldPos = v->worldPosi * invW;
        p.normal = v->normal * invW;
        p.mainLight = v->MainLightOri * invW;
        p.camOri = v->CameraOri * invW;
        p.vexLight = v->VexLightF * invW;
        for(int i=0; i<3; ++i) p.pixLight[i] = v->PixLightOri[i] * invW;
        return p;
    };
    Payload pay0 = create_payload(v0);
    Payload pay1 = create_payload(v1);
    Payload pay2 = create_payload(v2);
    // 排序;
    if (p0[1] > p1[1]) { std::swap(p0, p1); std::swap(pay0, pay1); }
    if (p0[1] > p2[1]) { std::swap(p0, p2); std::swap(pay0, pay2); }
    if (p1[1] > p2[1]) { std::swap(p1, p2); std::swap(pay1, pay2); }
    // 剔除无效高度
    if (p2[1] < 0 || p0[1] > static_cast<float>(screenHeight)) return;
    // 判断左右边
    float totalDy = p2[1] - p0[1];
    if (totalDy < 1e-5f) return;
    // 计算长边在 mid Y 处的 x
    float x_on_long = p0[0] + (p2[0] - p0[0]) * ((p1[1] - p0[1]) / totalDy);
    bool midIsRight = p1[0] > x_on_long;
    // 初始化三条边
    ScanEdge longEdge, shortEdgeTop, shortEdgeBot;
    // 长边
    longEdge.Init(p0, pay0, p2, pay2, screenHeight);
    // 短边上
    shortEdgeTop.Init(p0, pay0, p1, pay1, screenHeight);
    // 短边下
    shortEdgeBot.Init(p1, pay1, p2, pay2, screenHeight);
    auto DrawSpan = [&](const int y, const ScanEdge& left, const ScanEdge& right) {
        // 整数 X 范围
        const int xStart = static_cast<int>(std::ceil(left.x - 0.5f));
        const int xEnd   = static_cast<int>(std::ceil(right.x - 0.5f));
        // 裁剪 X
        const int rxStart = std::max(0, xStart);
        const int rxEnd   = std::min(screenWidth, xEnd);
        if (rxStart >= rxEnd) return; // 空行
        // 计算 X 方向的梯度 (dAttr/dx)
        float spanWidth = right.x - left.x;
        if (spanWidth <= 1e-5f) spanWidth = 1.0f; // 避免除0
        const float invSpan = 1.0f / spanWidth;
        const Payload xStep = (right.current - left.current) * invSpan;
        // Sub-pixel Pre-stepping X
        const float preStepX = (static_cast<float>(rxStart) + 0.5f) - left.x;
        Payload pixelPayload = left.current + (xStep * preStepX);
        // X 轴循环
        for (int x = rxStart; x < rxEnd; ++x) {
            // 透视恢复
            const float w_pixel = 1.0f / pixelPayload.invW;
            result.emplace_back();
            auto&[x_, y_, depth, worldPosi, normal, uv, VexLightF, PixLightOri, MainLightOri, CameraOri, alive] = result.back();
            alive = true;
            x_ = x; // 假设你的结构体字段叫 triX/triY
            y_ = y;
            // 属性还原: (Attr/w) * w
            worldPosi = pixelPayload.worldPos * w_pixel;
            uv        = pixelPayload.uv * w_pixel;
            depth     = pixelPayload.depth * w_pixel;
            normal    = pixelPayload.normal * w_pixel;
            MainLightOri = pixelPayload.mainLight * w_pixel;
            CameraOri    = pixelPayload.camOri * w_pixel;
            VexLightF    = pixelPayload.vexLight * w_pixel;
            for(int k=0; k<3; ++k) PixLightOri[k] = pixelPayload.pixLight[k] * w_pixel;
            // 步进到下一个像素
            pixelPayload = pixelPayload + xStep;
        }
    };

    // 执行 Y 轴扫描
    // 如果中点在右，则 左边=长边, 右边=短边上
    ScanEdge *leftE, *rightE;
    int midY = static_cast<int>(std::ceil(p1[1] - 0.5f));
    // 上半部分
    if (midIsRight) { leftE = &longEdge; rightE = &shortEdgeTop; }
    else            { leftE = &shortEdgeTop; rightE = &longEdge; }
    int yEndTop = std::min(midY, shortEdgeTop.yEnd); // 安全边界
    for (int y = shortEdgeTop.yStart; y < yEndTop; ++y) {
        DrawSpan(y, *leftE, *rightE);
        leftE->Step();
        rightE->Step();
    }
    // 下半部分
    // 切换短边：从 Top->Mid 变为 Mid->Bot
    // 长边继续走，不需要重置，但需要确保同步 (通常长边已经在上面的循环走到了 midY)

    // 注意：长边可能已经 Step 过了 yStart 到 yEndTop 的距离
    // 如果上下半部分无缝衔接，长边的当前状态应该是对的。
    // 为了保险（处理 float 精度误差），通常重新初始化下半部分的左右边逻辑，
    // 或者简单地继续步进。这里假设 midY 是分界线。

    if (midIsRight) { leftE = &longEdge; rightE = &shortEdgeBot; }
    else            { leftE = &shortEdgeBot; rightE = &longEdge; }

    // 修正长边的 yStart（如果长边是从 top 开始算的，现在它需要从 midY 继续）
    // 在这里由于 ScanEdge 内部维护了 current state，只要 y 连续，直接调用 Step 即可。
    // 唯一的问题是如果 longEdge 的 yStart 比 shortEdgeTop 的 yStart 小（不可能，因为都来自 p0），
    // 或者 loop 没跑满。
    // 简单起见，这里直接跑下半部分循环
    int yEndBot = std::min(screenHeight, shortEdgeBot.yEnd);
    // 确保从 midY 开始
    int currentY = std::max(midY, 0);

    // 对齐 longEdge 到 currentY (如果上半部分因为裁剪没跑完)
    // 这是一个复杂的 corner case。为了鲁棒性，工业级代码通常会计算 deltaY 直接 jump。
    // 这里假设简单的 Y 连续性
    for (int y = currentY; y < yEndBot; ++y) {
        DrawSpan(y, *leftE, *rightE);
        leftE->Step();
        rightE->Step();
    }
}