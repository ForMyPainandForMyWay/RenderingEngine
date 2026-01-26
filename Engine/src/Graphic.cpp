//
// Created by 冬榆 on 2025/12/31.
//

#include <iostream>
#include <random>
#include <ranges>

#include "Engine.h"
#include "F2P.h"
#include "FragTool.h"
#include "Mesh.h"
#include "Ray.hpp"
#include "RayTraceTool.hpp"
#include "HitInfo.hpp"
#include "RenderObjects.h"


Graphic::Graphic(Engine *eg) {
    this->engine = eg;
}

// 天空盒Pass
void Graphic::SkyPass(const SkyBox &obj,const Uniform &u, const GlobalUniform &gu, const int pass) const {
    const auto mesh = obj.getMesh();
    std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment> > FragMap;
    {
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle> > TriMap;
        VertexShading(TriMap, u, gu, mesh, pass);
        Clip(TriMap);
        ScreenMapping(TriMap, gu.getShadowViewPort()); // 注意是ShadowViewport,用的是Light的视窗参数
        Rasterization(TriMap, FragMap);
    }
    size_t count = 0;
    for (auto &Frag: FragMap | std::views::values) {
        count += Frag.size();
    }
    std::vector<F2P> result;
    result.reserve(count);
    FragmentShading(FragMap, result, u, pass);
    WriteBuffer(result);
}

// 阴影Pass
void Graphic::ShadowPass(const RenderObjects &obj,const Uniform &u, const GlobalUniform &gu, const int pass) const {
    const auto mesh = obj.getMesh();
    if (mesh == nullptr) return;
    if (mesh->getVBONums() == 0) return;
    std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment> > FragMap;
    {
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle> > TriMap;
        VertexShading(TriMap, u, gu, mesh, pass);
        Clip(TriMap);
        ScreenMapping(TriMap, gu.getShadowViewPort()); // 注意是ShadowViewport,用的是Light的视窗参数
        Rasterization(TriMap, FragMap);
    }
    size_t count = 0;
    for (auto &FragVec: FragMap | std::views::values) {
        Ztest(FragVec, engine->ShadowMap.ZBufferShadow);
        count += FragVec.size();
    }
    engine->ShadowMap.save();
}

// 基础纹理绘制,pass表示绘制层级
void Graphic::BasePass(const RenderObjects &obj,const Uniform &u, const GlobalUniform &gu, const int pass) const {
    const auto mesh = obj.getMesh();
    if (mesh == nullptr) return;
    if (mesh->getVBONums() == 0) return;
    // 顶点着色阶段
    /*
       更新：使用脏标记+Vector更好，不过需要注意
       在剔除比例较高时，考虑剔除时直接新建一个vector然后逐个将有效面移动过去
       这涉及到CPU的分支预测，后期可以进行优化
    */
    std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment>> FragMap;
    {
        std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> TriMap;
        auto start = std::chrono::high_resolution_clock::now();
        VertexShading(TriMap, u, gu, mesh, pass);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "顶点着色耗时: " << duration.count() << " 微秒\n";

        start = std::chrono::high_resolution_clock::now();
        // 完成顶点处理阶段后进行剔除、裁剪,最后齐次除法、面剔除
        Clip(TriMap);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "clip耗时: " << duration.count() << " 微秒\n";

        start = std::chrono::high_resolution_clock::now();
        // 退化检测、视口变换
        ScreenMapping(TriMap, gu.getScreenViewPort());
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "视口变换耗时: " << duration.count() << " 微秒\n";

        start = std::chrono::high_resolution_clock::now();
        // 几何着色
        GeometryShading(TriMap, u, mesh, pass);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "几何着色耗时: " << duration.count() << " 微秒\n";

        // 光栅化阶段，生成片元
        start = std::chrono::high_resolution_clock::now();
        Rasterization(TriMap, FragMap);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "光栅化耗时: " << duration.count() << " 微秒\n";
    }
    // 片段着色阶段，计算每个片元的颜色、光照和阴影处理
    // Early-Z,这里不清空ZBuffer，ZBuffer在每一帧的开始清空,由Engine控制
    size_t count = 0;
    for (auto &Frag: FragMap | std::views::values) {
        Ztest(Frag, engine->ZBuffer);
        count += Frag.size();
    }
    // 双重Z测试
    for (auto &Frag: FragMap | std::views::values) {
        Ztest(Frag, engine->ZBuffer);
        count += Frag.size();
    }
    std::vector<F2P> result;
    result.reserve(count);  // 预分配

    auto start = std::chrono::high_resolution_clock::now();
    // 基础颜色/纹理贴图采样(texture自动完成各向异性过滤和MipMap)
    FragmentShading(FragMap, result, u, pass);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "片元着色耗时: " << duration.count() << " 微秒\n";

    start = std::chrono::high_resolution_clock::now();
    // 写入GBuffer
    for (auto &Frag: FragMap | std::views::values) {
        WriteGBuffer(Frag);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "写入GBuffer耗时: " << duration.count() << " 微秒\n";

    start = std::chrono::high_resolution_clock::now();
    // 写入tmpBufferF
    WriteBuffer(result);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "写入tmpBuffer耗时: " << duration.count() << " 微秒\n";
    std::cout << std::endl;
}

void Graphic::RT(
    const std::vector<uint16_t>& models,
    const std::vector<RenderObjects>& renderObj,
    const uint8_t SPP,
    const uint8_t maxDepth) const {
    // 相机参数
    const float fovRad = engine->camera.getFov() * (3.1415926535f / 180.0f);
    const float Asp = engine->camera.getAspect();
    const float scale = std::tan(fovRad * 0.5f);

    const Vec3 cameraPos_ = engine->camera.getPosi();
    const Vec4 cameraPos{cameraPos_[0], cameraPos_[1], cameraPos_[2], 1.0f};
    const Mat4 cameraRot = engine->camera.RMat();

    const size_t width  = engine->width;
    const size_t height = engine->height;
    const size_t pixelCount = width * height;

    constexpr size_t BLOCK_SIZE = 1024;

    std::vector<std::future<std::vector<F2P>>> futures;
    futures.reserve((pixelCount + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (size_t i = 0; i < pixelCount; i += BLOCK_SIZE) {
        size_t start = i;
        size_t end   = std::min(i + BLOCK_SIZE, pixelCount);

        futures.emplace_back(engine->pool.addTask(
            [&, start, end, width, height, Asp, scale, SPP, maxDepth]() {
                std::vector<F2P> localResult;
                localResult.reserve(end - start);

                for (size_t idx = start; idx < end; ++idx) {
                    const size_t y = idx / width;
                    const size_t x = idx % width;
                    const float ndcX = (2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(width)) - 1.0f;
                    const float ndcY = 1.0f - (2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(height));
                    const float camX = ndcX * Asp * scale;
                    const float camY = ndcY * scale;
                    constexpr float camZ = -1.0f;
                    const Vec4 rayDirCamera{camX, camY, camZ, 0.0f};
                    const Vec4 rayDirWorld = normalize(cameraRot * rayDirCamera);
                    const Ray ray(cameraPos, rayDirWorld);
                    Vec3 radiance(0.0f);
                    // Path Tracing
                    for (int s = 0; s < SPP; ++s) {
                        Ray currentRay = ray;
                        Vec3 throughput(1.0f);
                        for (int depth = 0; depth < maxDepth; ++depth) {
                            const auto hitInfo = GetClosestHit(currentRay, models, renderObj);
                            if (!hitInfo) break;
                            const Material* material = hitInfo->material;
                            Vec3 hitAlbedo = BilinearSample(hitInfo->hitUV, material->KdMap).toFloat();
                            Vec3 hitEmission = Hadamard(material->Ke, hitAlbedo);
                            radiance += Hadamard(throughput, hitEmission);
                            if (depth >= 3) {  // 俄罗斯轮盘赌
                                if (float maxC = std::max({throughput[0], throughput[1], throughput[2]});
                                    maxC < 0.1f) {
                                    const float q = std::max(0.05f, maxC);
                                    if (RandomFloat() > q) break;
                                    throughput /= q;
                                }
                            }
                            // BSDF
                            Vec4 hitPos = currentRay.orignPosi + currentRay.Direction * hitInfo->t;
                            Vec4 nextDir = SampleCosineHemisphere(hitInfo->hitNormal);
                            if (dot(nextDir, hitInfo->hitNormal) < 0.0f) break;
                            throughput = Hadamard(throughput, hitAlbedo);
                            constexpr float EPS = 1e-4f;
                            currentRay.orignPosi = hitPos + hitInfo->hitNormal * EPS;
                            currentRay.Direction = nextDir;
                        }
                    }
                    radiance /= SPP;
                    localResult.push_back({
                        x,y,
                        {radiance[0], radiance[1], radiance[2]},
                        0.0f});
                }
                return localResult;
            }
        ));
    }
    // 结果合并
    std::vector<std::vector<F2P>> allParts;
    allParts.reserve(futures.size());
    size_t totalCount = 0;
    for (auto& f : futures) {
        auto part = f.get();
        totalCount += part.size();
        allParts.emplace_back(std::move(part));
    }
    std::vector<F2P> finalResult;
    finalResult.reserve(totalCount);
    for (auto& part : allParts) {
        finalResult.insert(
            finalResult.end(),
            std::make_move_iterator(part.begin()),
            std::make_move_iterator(part.end())
        );
    }
    WriteBuffer(finalResult);
}

void Graphic::WriteBuffer(const std::vector<F2P>& f2pVec) const {
    const size_t count = f2pVec.size();
    if (count == 0) return;

    // 如果像素数量很少（1万），直接主线程串行写完算了。
    const size_t width = engine->width; // 提前取出，避免多次指针解引用
    auto& buffer = engine->tmpBufferF;  // 获取引用

    if (constexpr size_t MIN_PARALLEL_SIZE = 10000;
        count < MIN_PARALLEL_SIZE) {
        for (const auto& f2p : f2pVec) {
            if (!f2p.alive) continue;
            // 注意：这里没有越界检查，假设上游逻辑保证了 x, y 合法
            buffer[f2p.x + f2p.y * width] = f2p.Albedo;
        }
        return;
    }
    constexpr size_t BLOCK_SIZE = 16384;
    std::vector<std::future<void>> futures;
    futures.reserve((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    for (size_t i = 0; i < count; i += BLOCK_SIZE) {
        size_t start = i;
        size_t end = std::min(i + BLOCK_SIZE, count);
        futures.emplace_back(engine->pool.addTask(
            [&f2pVec, &buffer, width, start, end]() {
                for (size_t k = start; k < end; ++k) {
                    const auto& f2p = f2pVec[k];
                    if (!f2p.alive) continue;
                    // 无数据竞争（同一坐标只有一个 alive fragment）
                    buffer[f2p.x + f2p.y * width] = f2p.Albedo;
                }
            }
        ));
    }
    for (auto& f : futures) f.get();
}

void Graphic::WriteGBuffer(const std::vector<Fragment>& f2pVec) const {
    const size_t count = f2pVec.size();
    if (count == 0) return;
    const size_t width = engine->width;
    auto& gData = engine->gBuffer->Gdata;
    constexpr size_t BLOCK_SIZE = 10240;
    std::vector<std::future<void>> futures;
    // 如果量太少，直接串行写，没必要启动线程池
    if (count < BLOCK_SIZE) {
        for (const auto& frag : f2pVec) {
            if (!frag.alive) continue;
            const auto i = frag.x + frag.y * width;
            gData[i].normal = frag.normal;
            gData[i].worldPosi = frag.worldPosi;
        }
        return;
    }
    // 启动线程池任务
    for (size_t i = 0; i < count; i += BLOCK_SIZE) {
        size_t start = i;
        size_t end = std::min(i + BLOCK_SIZE, count);
        futures.emplace_back(engine->pool.addTask(
            [&f2pVec, &gData, width, start, end]() {
                for (size_t k = start; k < end; ++k) {
                    const auto& frag = f2pVec[k];
                    if (!frag.alive) continue;  // 剔除的跳过
                    const auto idx = frag.x + frag.y * width;
                    // 无锁直接写入。不同的 alive frag 绝不会拥有相同的 x, y
                    gData[idx].normal = frag.normal;
                    gData[idx].worldPosi = frag.worldPosi;
                }
            }
        ));
    }
    for (auto& f : futures) f.get();
}