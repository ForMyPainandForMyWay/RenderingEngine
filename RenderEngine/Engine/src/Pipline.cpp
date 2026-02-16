//
// Created by 冬榆 on 2025/12/29.
//

#include <cmath>
#include <future>
#include <ranges>

#include "RenderObjects.hpp"
#include "ClipTool.hpp"
#include "Engine.hpp"
#include "F2P.hpp"
#include "MathTool.hpp"
#include "Mesh.hpp"
#include "RasterTool.hpp"
#include "BlinnShader.hpp"


// 应用阶段，对实例应用变换
void Engine::Application() {
    while (!tfCommand.empty()) {
        auto [objId, typeId, type, value] = tfCommand.front();
        tfCommand.pop();
        // 泛型 lambda：接受任意具有 updateP/Q/S 接口的对象
        auto applyCmd = [&](auto& obj) {
            switch (type) {
                case TRANSLATE: obj.updateP(value);break;
                case ROTATE: obj.updateQ(Euler2Quaternion(value));break;  // 注意转弧度制
                case SCALE: obj.updateS(value);break;
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
        size_t Min = 0xffffffff;
        size_t Max = 0;
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
            futures.emplace_back(engine->pool.addTask([&, start, end, Min] {
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
            futures.emplace_back(engine->pool.addTask([&triangles, start, end] {
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
            futures.emplace_back(engine->pool.addTask([&triangles, &ViewPort, start, end] {
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

void Graphic::GeometryShading(
    std::unordered_map<std::shared_ptr<Material>, std::vector<Triangle>>& TriMap,
    const Uniform &u, const std::shared_ptr<Mesh>& mesh, const int pass) const {

    const auto& PixLights = engine->PixLights;
    const auto& VexLights = engine->VexLights;
    const auto& mainLight = engine->mainLight;
    const auto& SdMap = engine->SdMap;
    const auto& envlight = engine->envLight;
    const auto& globalU = engine->globalU;

    // 遍历每个材质的三角形列表
    for (auto& [material, triangles] : TriMap) {
        const auto shader = material->getShader(pass);
        if (material->NormalMap == nullptr) {
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
                [&triangles, shader, material, start, end, PixLights, VexLights, mainLight, SdMap, envlight, globalU] {
                    for (size_t k = start; k < end; ++k) {
                        auto& tri = triangles[k];
                        if (!tri.alive) continue;
                        shader->GeometryShader(
                            tri,
                            material,
                            PixLights,
                            VexLights,
                            mainLight,
                            SdMap,
                            envlight,
                            globalU
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
                triFragCounts[i] = static_cast<uint32_t>(std::ceil(std::fabs(TriScreenArea2(tri))));
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
                            Scanline(tri, localFrags, width, height);
                             // BarycentricOptimizedFull(tri, localFrags, width, height);  // 这是重心坐标插值，性能差点
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
bool Graphic::ZTestPix(const size_t locate, const float depth, std::vector<float> &ZBuffer) {
    // 创建对 ZBuffer[locate] 的原子引用
    const std::atomic_ref zVal(ZBuffer[locate]);
    float oldZ = zVal.load(std::memory_order_relaxed);
    while (true) {
        if (oldZ < depth) {
            return false;
        }
        if (zVal.compare_exchange_weak(oldZ, depth, std::memory_order_relaxed)) {
            return true;
        }
    }
}

// ZTest传入Fragment
void Graphic::Ztest(std::vector<Fragment> &TestFrag, std::vector<float> &ZBuffer) const {
    const size_t count = TestFrag.size();
    if (count == 0) return;

    constexpr size_t BLOCK_SIZE = 8192;
    std::vector<std::future<int>> futures;
    futures.reserve((count + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (size_t i = 0; i < count; i += BLOCK_SIZE) {
        size_t start = i;
        size_t end = std::min(i + BLOCK_SIZE, count);

        // 提交任务，返回该块 kept 的数量
        futures.emplace_back(engine->pool.addTask(
            [&TestFrag, &ZBuffer, start, end, this] {
                int localKeptCount = 0;

                for (size_t k = start; k < end; ++k) {
                    auto& pix = TestFrag[k];
                    if (!pix.alive) continue;
                    if (const auto locate = pix.x + pix.y * static_cast<int>(this->engine->width);
                        ZTestPix(locate, pix.depth, ZBuffer)) {
                        pix.keep();
                        localKeptCount++;
                    } else pix.drop(); // 深度测试失败
                }
                return localKeptCount;
            }
        ));
    }
    int totalKept = 0;
    for (auto& f : futures) {
        totalKept += f.get();
    }
}

// 片元着色器
void Graphic::FragmentShading(
    const std::unordered_map<std::shared_ptr<Material>, std::vector<Fragment> >& fragMap,
    std::vector<F2P> &result, const Uniform &u, const int pass) const {

    const bool& NeedShadowPass = engine->settings[engine->renderSetting].NeedShadowPass;
    const auto& PixLights = engine->PixLights;
    const auto& mainLight = engine->mainLight;
    const auto& SdMap = engine->SdMap;
    const auto& envLight = engine->envLight;
    const auto& globalU = engine->globalU;

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
                [shader, material, &fragVec, start, end, SdMap, PixLights, mainLight, envLight, globalU, NeedShadowPass] {
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
                                PixLights,
                                mainLight,
                                SdMap,
                                envLight,
                                globalU,
                                NeedShadowPass
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
