//
// Created by 冬榆 on 2025/12/29.
//

#include <future>
#include <iostream>
#include <ranges>

#include "RenderObjects.h"
#include "ClipTool.h"
#include "Engine.h"
#include "F2P.h"
#include "MathTool.hpp"
#include "Mesh.h"
#include "RasterTool.hpp"
#include "BlinnShader.h"


// 应用阶段，对实例应用变换
void Engine::Application() {
    while (!tfCommand.empty()) {
        auto [objId, typeId, type, value] = tfCommand.front();
        tfCommand.pop();
        // 泛型 lambda：接受任意具有 updateP/Q/S 接口的对象
        auto applyCmd = [&](auto& obj) {
            switch (type) {
                case TfCmd::TRANSLATE: obj.updateP(value);break;
                case TfCmd::ROTATE: obj.updateQ(Euler2Quaternion(value));break;  // 注意转弧度制
                case TfCmd::SCALE: obj.updateS(value);break;
                default: break;
            }
        };
        if (typeId == CameraID) applyCmd(camera);         // camera 提供 updateP/Q/S
        else if (typeId == MainLightID && mainLight!= nullptr) applyCmd(*mainLight);  // light 提供相同接口
        else if (typeId >= PixL1 && typeId <= PixL3 && PixLights[typeId-PixL1].alive) applyCmd(PixLights[objId]);
        else if (typeId == RenderObject) applyCmd(renderObjs.at(objId));
        else if (typeId == VexLight) applyCmd(VexLights.at(objId));
        // else applyCmd(renderObjs.at(objId));  // renderObj 也提供相同接口
    }
}

// 顶点着色
// void Graphic::VertexShading(
//     std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>>& TriMap,
//     const Uniform &u, const GlobalUniform &gu, const std::shared_ptr<Mesh>& mesh, const int pass) {
//     auto VBO = mesh->VBO;
//     auto EBO = mesh->EBO;
//     for (const auto &sub : *mesh) {
//         auto material = sub.getMaterial();
//         shader = material->getShader(pass);
//         const auto oft = sub.getOffset();
//         const auto count = sub.getIdxCount();
//         const auto oftEnd = count + oft;
//         uint32_t Min = 0xffffffff;
//         uint32_t Max = 0;
//         std::vector<uint32_t> EBOcache;  // 顺序缓存subMesh的EBO索引
//         EBOcache.reserve(count);
//         for (auto idx = oft; idx < oftEnd; idx++) {
//             uint32_t id = EBO[idx];
//             EBOcache.push_back(id);
//             Min = id < Min ? id : Min;
//             Max = id > Max ? id : Max;
//         }
//         std::vector<V2F> vexList;  // 计算所有顶点并集中缓存
//         vexList.reserve(count);
//         for (auto idx = Min; idx <= Max; idx++) {
//             vexList.emplace_back(shader->VertexShader(VBO[idx], u, gu));
//         }
//
//         auto& triangleList = TriMap[material];
//         triangleList.reserve(triangleList.size() + count / 3);  // 预分配内存
//         TriMap[material].reserve((oftEnd-oft-1) / 3);
//         for (auto idx = oft; idx < oftEnd; idx+=3) {
//             V2F v1 =  vexList[EBOcache[idx]];
//             V2F v2 =  vexList[EBOcache[idx+1]];
//             V2F v3 =  vexList[EBOcache[idx+2]];
//             TriMap[material].emplace_back(Triangle{{v1, v2, v3}});
//         }
//     }
// }
void Graphic::VertexShading(
    std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>>& TriMap,
    const Uniform &u, const GlobalUniform &gu, const std::shared_ptr<Mesh>& mesh, const int pass) const {

    const auto& VBO = mesh->VBO;
    const auto& EBO = mesh->EBO;
    for (const auto &sub : *mesh) {
        auto material = sub.getMaterial();
        auto shader = material->getShader(pass);
        const auto oft = sub.getOffset();
        const auto count = sub.getIdxCount();
        const auto oftEnd = count + oft;
        uint32_t Min = 0xffffffff;
        uint32_t Max = 0;
        // 预处理 EBO 和 Min/Max (串行)
        std::vector<uint32_t> EBOcache;
        EBOcache.reserve(count);
        for (auto idx = oft; idx < oftEnd; idx++) {
            uint32_t id = EBO[idx];
            EBOcache.push_back(id);
            Min = id < Min ? id : Min;
            Max = id > Max ? id : Max;
        }
        // 访问时必须用 (id - Min) 作为下标，否则 vexList[EBOcache[idx]] 会越界。
        size_t vertexCount = Max - Min + 1;
        std::vector<V2F> vexList;
        vexList.resize(vertexCount); // 必须预分配内存，否则多线程 push_back 会崩溃
        constexpr size_t BLOCK_SIZE = 400; // 每个任务处理的顶点数
        std::vector<std::future<void>> futures;
        // 按块拆分任务
        for (size_t i = 0; i < vertexCount; i += BLOCK_SIZE) {
            size_t start = i;
            size_t end = std::min(i + BLOCK_SIZE, vertexCount);
            // 提交任务到线程池
            // 按值捕获基础类型，按引用捕获 vexList/VBO 等大对象
            futures.emplace_back(engine->pool.addTask([&, start, end, Min]() {
                for (size_t k = start; k < end; k++) {
                    // 计算绝对索引
                    const uint32_t vboIndex = Min + k;
                    // 直接写入预分配好的位置，不同线程写入不同 k，无需加锁
                    vexList[k] = shader->VertexShader(VBO[vboIndex], u, gu);
                }
            }));
        }
        // 等待所有顶点计算完成
        for (auto &f : futures) f.get(); // 阻塞直到该块完成
        // 组装三角形 (保持串行)
        auto& triangleList = TriMap[material];
        triangleList.reserve(triangleList.size() + count / 3);
        // 访问 vexList 需减去 Min
        for (size_t idx = 0; idx < EBOcache.size(); idx += 3) {
            // 这里假设 EBOcache 存的是原始 VBO 索引
            V2F v1 = vexList[EBOcache[idx]   - Min];
            V2F v2 = vexList[EBOcache[idx+1] - Min];
            V2F v3 = vexList[EBOcache[idx+2] - Min];
            triangleList.emplace_back(Triangle{{(v1), (v2), (v3)}});
        }
    }
}
// 顶点着色后处理: 视锥剔除、SH裁剪、透视除法、背面剔除、深度映射
// void Graphic::Clip(std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &map) const{
//     for (auto& triangles: map | std::views::values) {
//         // 三点组成一个三角形
//         std::vector<Triangle> result;
//         result.reserve(triangles.size());  // 预分配一倍内存
//         for (auto& triangle : triangles) {
//             V2F &p1 = triangle[0];
//             V2F &p2 = triangle[1];
//             V2F &p3 = triangle[2];
//             // 快速剔除全部在外的
//             if (AllVertexOutside(p1, p2, p3)) {
//                 triangle.alive = false;
//                 continue;
//             }
//             std::vector<Triangle> tris;
//             // 裁剪
//             const bool clip = !AllVertexInside(p1, p2, p3);
//             if (clip) {
//                 triangle.alive = false;  // 先剔除原来的旧三角
//                 tris = PolyClip(p1, p2, p3);  // S-H算法.直接返回切分后的三角数组
//             }
//             if (clip) {
//                 for (auto &t : tris) {
//                     PersDiv(t);   // 透视除法->NDC空间
//                     FaceClip(t);  // 背面剔除
//                     // DepthMap(t);  // 深度映射
//                     result.emplace_back(t);
//                 }
//             } else {
//                 PersDiv(triangle);
//                 FaceClip(triangle);
//                 // DepthMap(triangle);
//             }
//         }
//         triangles.insert(triangles.end(),
//                 result.begin(), result.end());
//     }
// }
void Graphic::Clip(std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &map) const{
    // 遍历每一个材质的三角形列表
    for (auto& triangles: map | std::views::values) {
        size_t count = triangles.size();
        if (count == 0) continue;
        constexpr size_t BLOCK_SIZE = 512;
        std::vector<std::future<std::vector<Triangle>>> futures;
        futures.reserve((count + BLOCK_SIZE - 1) / BLOCK_SIZE); // 预留 future 空间
        for (size_t i = 0; i < count; i += BLOCK_SIZE) {
            size_t start = i;
            size_t end = std::min(i + BLOCK_SIZE, count);
            futures.emplace_back(engine->pool.addTask([&triangles, start, end]() {
                std::vector<Triangle> localResult;
                // 假设 10% 发生裁剪产生新三角
                localResult.reserve((end - start) / 10);
                for (size_t k = start; k < end; ++k) {
                    auto& triangle = triangles[k]; // 引用访问，无数据竞争
                    V2F &p1 = triangle[0];
                    V2F &p2 = triangle[1];
                    V2F &p3 = triangle[2];
                    if (AllVertexOutside(p1, p2, p3)) {
                        triangle.alive = false;  // 快速剔除全都在外的
                        continue;
                    }
                    if (!AllVertexInside(p1, p2, p3)) {
                        triangle.alive = false; // 需要裁剪
                        // 裁剪产生新三角形
                        for (auto tris = PolyClip(p1, p2, p3); auto &t : tris) {
                            PersDiv(t);
                            FaceClip(t);
                            // DepthMap(t);
                            localResult.emplace_back(t);
                        }
                    } else {
                        // 原地修改
                        PersDiv(triangle);
                        FaceClip(triangle);
                        // DepthMap(triangle);
                    }
                }
                return localResult; // 移动语义返回
            }));
        }
        // 临时存储所有子线程产生的结果列表，避免反复 get()
        std::vector<std::vector<Triangle>> allNewParts;
        allNewParts.reserve(futures.size());
        size_t totalNewTriangles = 0;
        // 等待所有线程完成，并统计新增三角形总数
        for (auto &f : futures) {
            if (auto part = f.get(); !part.empty()) {
                totalNewTriangles += part.size();
                // 暂时存起来，所有权转移到 allNewParts 中
                allNewParts.emplace_back(std::move(part));
            }
        }
        // 如果有新增三角形，执行一次性扩容和搬运
        if (totalNewTriangles > 0) {
            // 这避免了多次 insert 导致的多次 realloc 和全量数据拷贝
            triangles.reserve(triangles.size() + totalNewTriangles);
            // 批量搬运
            for (auto &part : allNewParts) {
                triangles.insert(
                    triangles.end(),
                    std::make_move_iterator(part.begin()), // 强制移动元素，而非复制
                    std::make_move_iterator(part.end())
                );
            }
        }
    }
}

// 视口变换把坐标从NDC转换到Screen、面积退化检测
// void Graphic::ScreenMapping(std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &map, const Mat4&ViewPort) {
//     for (auto& triangles : map | std::views::values) {
//         for (auto& tri : triangles) {
//             if (!tri.alive) continue;
//             tri[0].clipPosi = ViewPort * tri[0].clipPosi;
//             tri[1].clipPosi = ViewPort * tri[1].clipPosi;
//             tri[2].clipPosi = ViewPort * tri[2].clipPosi;
//             auto clampCoord = [&](Vec4& p) {
//                 if (p[0] < 0 && p[0] > -1e-4) p[0] = 0.0f;
//                 if (p[1] < 0 && p[1] > -1e-4) p[1] = 0.0f;
//             };  // 舍入处理
//             clampCoord(tri[0].clipPosi);
//             clampCoord(tri[1].clipPosi);
//             clampCoord(tri[2].clipPosi);
//             DegenerateClip(tri);  // 退化检测(面积过小的三角)
//         }
//     }
// }
void Graphic::ScreenMapping(std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &map, const Mat4& ViewPort) const {
    // 遍历每个材质的三角形列表
    for (auto& triangles : map | std::views::values) {
        size_t count = triangles.size();
        if (count == 0) continue;
        constexpr size_t BLOCK_SIZE = 1024;
        std::vector<std::future<void>> futures;
        // 提交并行任务
        for (size_t i = 0; i < count; i += BLOCK_SIZE) {
            size_t start = i;
            size_t end = std::min(i + BLOCK_SIZE, count);
            // 捕获 ViewPort和 triangles
            futures.emplace_back(engine->pool.addTask([&triangles, &ViewPort, start, end]() {
                // 为了性能，将 lambda 放在任务内部
                auto clampCoord = [](Vec4& p) {
                    // 这里去掉了 -1e-4 的判断，通常直接判断 < 0 即可，
                    // 但如果你有特殊的 epsilon 需求，保持原样即可。
                    if (p[0] < 0 && p[0] > -1e-4) p[0] = 0.0f;
                    if (p[1] < 0 && p[1] > -1e-4) p[1] = 0.0f;
                };
                for (size_t k = start; k < end; ++k) {
                    auto& tri = triangles[k];
                    // 被剔除
                    if (!tri.alive) continue;
                    tri[0].clipPosi = ViewPort * tri[0].clipPosi;
                    tri[1].clipPosi = ViewPort * tri[1].clipPosi;
                    tri[2].clipPosi = ViewPort * tri[2].clipPosi;
                    clampCoord(tri[0].clipPosi);
                    clampCoord(tri[1].clipPosi);
                    clampCoord(tri[2].clipPosi);
                    // 面积退化检测
                    DegenerateClip(tri);
                }
            }));
        }
        for (auto& f : futures) f.get();
    }
}

// void Graphic::GeometryShading(
//     std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>>& TriMap,
//     const Uniform &u, const std::shared_ptr<Mesh>& mesh, const int pass) {
//     for (auto& [material, triangles] : TriMap) {
//         shader = material->getShader(pass);
//         if (material->BumpMap != nullptr) {
//             for (auto& tri : triangles) {
//                 if (!tri.alive) continue;
//                 shader->GeometryShader(
//                 tri,material,
//                 engine->PixLights,
//                 engine->VexLights,
//                 engine->mainLight,
//                 engine->ShadowMap,
//                 engine->envLight,
//                 engine->globalU);
//             }
//         }
//     }
// }
void Graphic::GeometryShading(
    std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>>& TriMap,
    const Uniform &u, const std::shared_ptr<Mesh>& mesh, const int pass) const {
    // 遍历每个材质的三角形列表
    for (auto& [material, triangles] : TriMap) {
        const auto shader = material->getShader(pass);
        if (material->BumpMap == nullptr) {
            continue;
        }
        size_t count = triangles.size();
        if (count == 0) continue;
        constexpr size_t BLOCK_SIZE = 512;
        std::vector<std::future<void>> futures;
        futures.reserve(triangles.size());
        for (size_t i = 0; i < count; i += BLOCK_SIZE) {
            const size_t start = i;
            const size_t end = std::min(i + BLOCK_SIZE, count);
            futures.emplace_back(engine->pool.addTask(
                [&triangles, shader, material, start, end, this]() {
                    for (size_t k = start; k < end; ++k) {
                        auto& tri = triangles[k];
                        if (!tri.alive) continue;
                        shader->GeometryShader(
                            tri,
                            material,
                            this->engine->PixLights,
                            this->engine->VexLights,
                            this->engine->mainLight,
                            this->engine->ShadowMap,
                            this->engine->envLight,
                            this->engine->globalU
                        );
                    }
                }
            ));
        }
        for (auto& f : futures) {
            f.get();
        }
    }
}

// 光栅化接口
// void Graphic::Rasterization(
    // std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &TriMap,
    // std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment>> &FragMap) const {
    // const int width = static_cast<int>(engine->width);
    // const int height = static_cast<int>(engine->height);
    // for (auto& [material, triangles] : TriMap) {
        // std::vector<Fragment> fragVec;
        // size_t area = triangles.size()*10;  // 预分配偏移量
        // for (auto& tri : triangles) {
            // area += static_cast<size_t>(ceil(fabs(TriScreenArea2(tri))));
        // }
        // fragVec.reserve(area);
        // for (auto& tri : triangles) {
            // if (!tri.alive) continue;  // 背面剔除、退化剔除
            // 光栅化并返回该三角形的片元序列
            // sortTriangle(tri);
            // std::vector<Fragment> triFrags;  // 在光栅化内部预分配线程缓冲区内存
            // BarycentricOptimizedFull(tri, triFrags, width, height);
            // 加入序列到fragVec中
            // fragVec.insert(fragVec.end(), std::make_move_iterator(triFrags.begin()), std::make_move_iterator(triFrags.end()));
        // }
        // fragVec.shrink_to_fit();  // 清空无效内存
        // 设置片元序列
        // FragMap.emplace(material, std::move(fragVec));
    // }
// }
void Graphic::Rasterization(
    std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>> &TriMap,
    std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment>> &FragMap) const {
    const int width = static_cast<int>(engine->width);
    const int height = static_cast<int>(engine->height);
    for (auto& [material, triangles] : TriMap) {
        size_t triCount = triangles.size();
        if (triCount == 0) continue;

        // 预计算面积缓存
        // 存储每个三角形预估产生的片元数量
        std::vector<uint32_t> triFragCounts;
        triFragCounts.resize(triCount);
        for (size_t i = 0; i < triCount; ++i) {
            if (const auto& tri = triangles[i]; tri.alive) {
                // 计算该三角形的片元数估算值
                triFragCounts[i] = static_cast<uint32_t>(ceil(fabs(TriScreenArea2(tri))));
            } else triFragCounts[i] = 0;
        }
        constexpr size_t BLOCK_SIZE = 512;
        std::vector<std::future<std::vector<Fragment>>> futures;

        // 预分配 future 空间
        futures.reserve((triCount + BLOCK_SIZE - 1) / BLOCK_SIZE);
        for (size_t i = 0; i < triCount; i += BLOCK_SIZE) {
            size_t start = i;
            size_t end = std::min(i + BLOCK_SIZE, triCount);
            futures.emplace_back(engine->pool.addTask(
                [&triangles, &triFragCounts, start, end, width, height] {
                    // 计算当前块需要的精确内存大小
                    size_t exactLocalSize = 0;
                    for (size_t k = start; k < end; ++k) {
                        exactLocalSize += triFragCounts[k];
                    }
                    std::vector<Fragment> localFrags;
                    localFrags.reserve(exactLocalSize + 10*(end-start));
                    for (size_t k = start; k < end; ++k) {
                        if (auto& tri = triangles[k]; tri.alive) { // 利用 cached count 判断 alive
                             BarycentricOptimizedFull(tri, localFrags, width, height);
                        }
                    }
                    return localFrags;
                }
            ));
        }
        // 结果合并
        std::vector<std::vector<Fragment>> allParts;
        allParts.reserve(futures.size());
        size_t totalExactCount = 0;
        for (auto& f : futures) {
            std::vector<Fragment> part = f.get();
            totalExactCount += part.size();
            allParts.emplace_back(std::move(part));
        }
        std::vector<Fragment> finalFrags;
        finalFrags.reserve(totalExactCount);
        for (auto& part : allParts) {
            finalFrags.insert(
                finalFrags.end(),
                std::make_move_iterator(part.begin()),
                std::make_move_iterator(part.end())
            );
        }
        FragMap.emplace(material, std::move(finalFrags));
    }
}

// ZTest组件
// bool Graphic::ZTestPix(const size_t locate, const float depth, std::vector<float> &ZBuffer) {
//     if (ZBuffer[locate] >= depth) {
//         ZBuffer[locate] = depth;
//         return true;
//     }
//     return false;
// }
// 修改 Graphic::ZTestPix 函数，或者新建一个 ThreadSafeZTestPix
bool Graphic::ZTestPix(const size_t locate, const float depth, std::vector<float> &ZBuffer) {
    // 1. 创建对 ZBuffer[locate] 的原子引用
    // 这不会拷贝数据，只是把普通 float 当作 atomic 来操作
    const std::atomic_ref zVal(ZBuffer[locate]);

    float oldZ = zVal.load(std::memory_order_relaxed);

    // 2. CAS 循环 (Compare-And-Swap)
    while (true) {
        // 如果新深度比旧深度大（更远），直接失败，不需要更新
        if (oldZ < depth) {
            return false;
        }

        // 尝试把 oldZ 替换为 depth
        // 如果 zVal 依然等于 oldZ，则替换并返回 true
        // 如果 zVal 被其他线程改了（不等于 oldZ），则把 zVal 的新值赋给 oldZ，并返回 false (循环重试)
        if (zVal.compare_exchange_weak(oldZ, depth, std::memory_order_relaxed)) {
            return true;
        }
        // 如果走到这里，说明 compare_exchange_weak 失败了，oldZ 已经被更新为最新值
        // 循环会继续，再次判断 if (oldZ < depth)
    }
}

// ZTest传入Fragment
// void Graphic::Ztest(std::vector<Fragment> &TestFrag, std::vector<float> &ZBuffer) const {
//     int keptCount = 0;
//     for (auto &pix : TestFrag) {
//         if (!pix.alive) continue;
//         if (const auto locate = pix.x + pix.y * engine->width;
//             ZTestPix(locate, pix.depth, ZBuffer)) {
//             pix.keep();
//             keptCount++;
//         }
//         else pix.drop();  // 后续frag不再着色
//     }
// }
void Graphic::Ztest(std::vector<Fragment> &TestFrag, std::vector<float> &ZBuffer) const {
    const size_t count = TestFrag.size();
    if (count == 0) return;

    // 1. 分块并行
    // ------------------------------------------------
    // ZTest 计算量极小（基本是访存），所以 BLOCK_SIZE 要很大，避免调度开销
    constexpr size_t BLOCK_SIZE = 8192;

    std::vector<std::future<int>> futures;
    futures.reserve((count + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (size_t i = 0; i < count; i += BLOCK_SIZE) {
        size_t start = i;
        size_t end = std::min(i + BLOCK_SIZE, count);

        // 提交任务，返回该块 kept 的数量
        futures.emplace_back(engine->pool.addTask(
            [&TestFrag, &ZBuffer, start, end, this]() {
                int localKeptCount = 0;

                for (size_t k = start; k < end; ++k) {
                    auto& pix = TestFrag[k];

                    if (!pix.alive) continue;

                    // 计算屏幕索引
                    // const auto locate = pix.x + pix.y * engine->width;
                    // 注意：这里需要确保 x, y 没越界，虽然通常光栅化阶段保证了

                    // 使用原子操作进行 ZTest
                    // 如果通过测试，AtomicMinDepth 会自动更新 ZBuffer
                    if (const uint32_t locate = pix.x + pix.y * static_cast<int>(this->engine->width); ZTestPix(locate, pix.depth, ZBuffer)) {
                        pix.keep();
                        localKeptCount++;
                    } else {
                        pix.drop(); // 深度测试失败
                    }
                }
                return localKeptCount;
            }
        ));
    }

    // 2. 汇总结果 (Reduce)
    // ------------------------------------------------
    int totalKept = 0;
    for (auto& f : futures) {
        totalKept += f.get();
    }

    // 如果你需要 totalKept 做统计的话可以使用，否则不需要返回
    // printf("ZTest Passed: %d\n", totalKept);
}

// 片元着色器
// void Graphic::FragmentShading(
//     const std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment> >& fragMap,
//     std::vector<F2P> &result, const Uniform &u, const int pass) const {
//     for (auto& [material, fragVec] : fragMap) {
//         const auto shader = material->getShader(pass);
//         for (auto& frag : fragVec) {
//             if (!frag.alive) continue;
//             result.emplace_back(
//                 shader->FragmentShader(
//                 frag,material,
//                 engine->PixLights,
//                 engine->mainLight,
//                 engine->ShadowMap,
//                 engine->envLight,
//                 engine->globalU,
//                 engine->NeedShadowPass));
//         }
//     }
// }
void Graphic::FragmentShading(
    const std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment> >& fragMap,
    std::vector<F2P> &result, const Uniform &u, const int pass) const {

    std::vector<std::future<std::vector<F2P>>> futures;
    //  遍历材质 Fragment 列表并分发任务
    for (const auto& [material, fragVec] : fragMap) {
        size_t count = fragVec.size();
        if (count == 0) continue;
        const auto shader = material->getShader(pass);
        // 确定块大小
        constexpr size_t BLOCK_SIZE = 1024;
        for (size_t i = 0; i < count; i += BLOCK_SIZE) {
            const size_t start = i;
            const size_t end = std::min(i + BLOCK_SIZE, count);
            // 提交任务
            futures.emplace_back(engine->pool.addTask(
                [shader, material, &fragVec, start, end, this]() {
                    // A. 线程局部存储
                    std::vector<F2P> localResult;
                    // 预分配内存：假设大部分 fragment 都是 alive 的
                    localResult.reserve(end - start);
                    for (size_t k = start; k < end; ++k) {
                        const auto& frag = fragVec[k];
                        if (!frag.alive) continue;
                        localResult.emplace_back(
                            shader->FragmentShader(
                                frag,
                                material,
                                this->engine->PixLights,
                                this->engine->mainLight,
                                this->engine->ShadowMap,
                                this->engine->envLight,
                                this->engine->globalU,
                                this->engine->NeedShadowPass
                            )
                        );
                    }
                    return localResult; // 移动语义返回
                }
            ));
        }
    }
    // 预先收集所有局部结果，统计总数
    std::vector<std::vector<F2P>> allParts;
    allParts.reserve(futures.size());
    size_t totalCount = 0;
    for (auto& f : futures) {
        std::vector<F2P> part = f.get();
        totalCount += part.size();
        allParts.emplace_back(std::move(part));
    }
    // 批量搬运,result已经在外部扩容过
    for (auto& part : allParts) {
        result.insert(
            result.end(),
            std::make_move_iterator(part.begin()),
            std::make_move_iterator(part.end())
        );
    }
}

// // Lately-Z,传入F2P,不能清除ZBuffer
// void Graphic::Ztest(std::vector<F2P> &TestPix, std::vector<float> &ZBuffer) const {
//     int keptCount = 0;
//     for (auto &pix : TestPix) {
//         if (const auto locate = pix.x + pix.y * engine->width;
//             ZTestPix(locate, pix.depth, ZBuffer)) {
//             pix.keep();
//             keptCount++;
//             }
//         else pix.drop();  // 后续frag不再着色
//     }
// }

