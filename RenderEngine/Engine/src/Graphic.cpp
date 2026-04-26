//
// Created by 冬榆 on 2025/12/31.
//

#include <cmath>
#include <future>
#include <ranges>

#include "Engine.hpp"
#include "F2P.hpp"
#include "FragTool.hpp"
#include "Mesh.hpp"
#include "Ray.hpp"
#include "RayTraceTool.hpp"
#include "Shader.hpp"
#include "HitInfo.hpp"
#include "RenderObjects.hpp"
#include "ClipTool.hpp"
#include "RasterTool.hpp"
#include "MathTool.hpp"
#if USE_CUDA
#include "BVH.cuh"
#include "DATAPackegGPU.cuh"
#include "Interface.cuh"
#include "TransfomTool.cuh"
#include "Mesh.cuh"
#include "Shape.cuh"
#endif


Graphic::Graphic(Engine *eg) {
    this->engine = eg;
}

// 天空盒Pass
void Graphic::SkyPass(const SkyBox &obj,const Uniform &u, const GlobalUniform &gu, const int pass) const {
    const auto mesh = obj.getMesh();
    if (mesh == nullptr) return;
    if (mesh->getVBONums() == 0) return;

    const auto& VBO = mesh->VBO;
    const auto& EBO = mesh->EBO;
    const auto& shadowViewPort = gu.getShadowViewPort();

    for (const auto &sub : *mesh) {
        auto material = sub.getMaterial();
        const auto oft = sub.getOffset();
        const auto count = sub.getIdxCount();
        const auto oftEnd = count + oft;

        std::vector<uint32_t> EBOcache;
        EBOcache.reserve(count);
        for (auto idx = oft; idx < oftEnd; idx++) {
            EBOcache.push_back(EBO[idx]);
        }

        constexpr size_t BLOCK_SIZE = 16;
        std::vector<std::future<void>> futures;
        futures.reserve(EBOcache.size() / 3 / BLOCK_SIZE + 1);
        for (size_t i = 0; i < EBOcache.size(); i += 3 * BLOCK_SIZE) {
            const size_t start = i;
            const size_t end = std::min(i + 3 * BLOCK_SIZE, EBOcache.size());
            futures.emplace_back(engine->pool.addTask([&, start, end] {
                for (size_t idx = start; idx < end; idx += 3) {
                    ProcessTriangle(idx, EBOcache, VBO, material, u, gu, pass, shadowViewPort);
                }
            }));
        }
        for (auto& f : futures) f.get();
    }
}

// 阴影Pass
void Graphic::ShadowPass(const RenderObjects &obj,const Uniform &u, const GlobalUniform &gu, const int pass) const {
    const auto mesh = obj.getMesh();
    if (mesh == nullptr) return;
    if (mesh->getVBONums() == 0) return;

    const auto& VBO = mesh->VBO;
    const auto& EBO = mesh->EBO;
    const auto& shadowViewPort = gu.getShadowViewPort();

    for (const auto &sub : *mesh) {
        auto material = sub.getMaterial();
        const auto oft = sub.getOffset();
        const auto count = sub.getIdxCount();
        const auto oftEnd = count + oft;

        std::vector<uint32_t> EBOcache;
        EBOcache.reserve(count);
        for (auto idx = oft; idx < oftEnd; idx++) {
            EBOcache.push_back(EBO[idx]);
        }

        constexpr size_t BLOCK_SIZE = 16;
        std::vector<std::future<void>> futures;
        futures.reserve(EBOcache.size() / 3 / BLOCK_SIZE + 1);
        for (size_t i = 0; i < EBOcache.size(); i += 3 * BLOCK_SIZE) {
            const size_t start = i;
            const size_t end = std::min(i + 3 * BLOCK_SIZE, EBOcache.size());
            futures.emplace_back(engine->pool.addTask([&, start, end] {
                for (size_t idx = start; idx < end; idx += 3) {
                    ProcessTriangle(idx, EBOcache, VBO, material, u, gu, pass, shadowViewPort, true);
                }
            }));
        }
        for (auto& f : futures) f.get();
    }
    // engine->SdMap->save();  // 将shadowMap写入文件
}

// 基础纹理绘制,pass表示绘制层级
void Graphic::BasePass(const RenderObjects &obj,const Uniform &u, const GlobalUniform &gu, const int pass) const {
    const auto mesh = obj.getMesh();
    if (mesh == nullptr) return;
    if (mesh->getVBONums() == 0) return;
    const auto& VBO = mesh->VBO;
    const auto& EBO = mesh->EBO;
    const auto& screenViewPort = gu.getScreenViewPort();

    for (const auto &sub : *mesh) {
        auto material = sub.getMaterial();
        const auto oft = sub.getOffset();
        const auto count = sub.getIdxCount();
        const auto oftEnd = count + oft;

        std::vector<uint32_t> EBOcache;
        EBOcache.reserve(count);
        for (auto idx = oft; idx < oftEnd; idx++) {
            EBOcache.push_back(EBO[idx]);
        }

        constexpr size_t BLOCK_SIZE = 16;
        std::vector<std::future<void>> futures;
        futures.reserve(EBOcache.size() / 3 / BLOCK_SIZE + 1);
        for (size_t i = 0; i < EBOcache.size(); i += 3 * BLOCK_SIZE) {
            const size_t start = i;
            const size_t end = std::min(i + 3 * BLOCK_SIZE, EBOcache.size());
            futures.emplace_back(engine->pool.addTask([&, start, end] {
                for (size_t idx = start; idx < end; idx += 3) {
                    ProcessTriangle(idx, EBOcache, VBO, material, u, gu, pass, screenViewPort);
                }
            }));
        }
        for (auto& f : futures) f.get();
    }
}

void Graphic::ProcessTriangle(
    const size_t idx,
    const std::vector<uint32_t>& EBOcache,
    const std::vector<Vertex>& VBO,
    const std::shared_ptr<Material>& material,
    const Uniform& u,
    const GlobalUniform& gu,
    const int pass,
    const Mat4& viewPort,
    const bool shadowPass) const {

    if (material == nullptr) return;
    const auto shader = material->getShader(pass);
    if (shader == nullptr) return;

    V2F v1 = shader->VertexShader(VBO[EBOcache[idx]], u, gu);
    V2F v2 = shader->VertexShader(VBO[EBOcache[idx + 1]], u, gu);
    V2F v3 = shader->VertexShader(VBO[EBOcache[idx + 2]], u, gu);
    Triangle triangle{{v1, v2, v3}};

    if (AllVertexOutside(triangle[0], triangle[1], triangle[2])) return;

    auto screenMap = [&viewPort](Triangle& tri) {
        auto clampCoord = [](Vec4& p) {
            if (p[0] < 0 && p[0] > -1e-4) p[0] = 0.0f;
            if (p[1] < 0 && p[1] > -1e-4) p[1] = 0.0f;
        };
        tri[0].clipPosi = viewPort * tri[0].clipPosi;
        tri[1].clipPosi = viewPort * tri[1].clipPosi;
        tri[2].clipPosi = viewPort * tri[2].clipPosi;
        clampCoord(tri[0].clipPosi);
        clampCoord(tri[1].clipPosi);
        clampCoord(tri[2].clipPosi);
        DegenerateClip(tri);
    };

    auto rasterize = [&](Triangle& tri) {
        if (!tri.alive) return;
        if (shadowPass && engine->SdMap == nullptr) return;

        std::vector<Fragment> fragments;
        fragments.reserve(static_cast<size_t>(std::ceil(std::fabs(TriScreenArea2(tri)))) + 8);

        const int width = shadowPass ? static_cast<int>(engine->SdMap->width) : static_cast<int>(engine->width);
        const int height = shadowPass ? static_cast<int>(engine->SdMap->height) : static_cast<int>(engine->height);
        Scanline(tri, fragments, width, height);

        if (shadowPass) {
            auto& zbuf = engine->SdMap->ZBufferShadow;
            const size_t zWidth = engine->SdMap->width;
            for (auto& frag : fragments) {
                if (!frag.alive) continue;
                if (const size_t locate = frag.x + frag.y * zWidth;
                    ZTestPix(locate, frag.depth, zbuf)) frag.keep();
                else frag.drop();
            }
            return;
        }

        const bool& NeedShadowPass = engine->settings[0].NeedShadowPass;
        const auto& PixLights = engine->PixLights;
        const auto& mainLight = engine->mainLight;
        const auto& SdMap = engine->SdMap;
        const auto& envLight = engine->envLight;
        const auto& globalU = engine->globalU;

        for (auto& frag : fragments) {
            if (!Ztest(frag, engine->ZBuffer)) continue;
            WriteGBuffer(frag);
            const F2P f2p = shader->FragmentShader(
                frag, material, PixLights, mainLight, SdMap, envLight, globalU, NeedShadowPass);
            WriteBuffer(f2p);
        }
    };

    if (!AllVertexInside(triangle[0], triangle[1], triangle[2])) {
        for (auto tris = PolyClip(triangle[0], triangle[1], triangle[2]); auto &t : tris) {
            PersDiv(t);
            FaceClip(t);
            if (!t.alive) continue;
            screenMap(t);
            if (!t.alive) continue;
            GeometryShading(t, material, u, pass);
            if (!t.alive) continue;
            rasterize(t);
        }
        return;
    }

    PersDiv(triangle);
    FaceClip(triangle);
    if (!triangle.alive) return;
    screenMap(triangle);
    if (!triangle.alive) return;
    GeometryShading(triangle, material, u, pass);
    if (!triangle.alive) return;
    rasterize(triangle);
}

void Graphic::RT(
    const uint8_t SSP,
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

    // 最终结果容器
    std::vector<F2P> finalResult;
    finalResult.reserve(pixelCount); // 预分配避免重复扩容

    #if USE_CUDA
    // 这里进入CUDA逻辑，先把数据打包好，然后送入接口函数
    // 打包相机数据
    CameraDataGPU cameraDataGPU{
        Asp, scale, Vec4ToFloat4(cameraPos), Mas4ToGPU(cameraRot), width, height, pixelCount};

    // 构建纹理贴图映射表
    int textNums = 0;
    std::unordered_map<std::shared_ptr<TextureMap>, int> TextMap;
    for (const auto& text : engine->textureMap | std::views::values) {
        TextMap[text] = textNums++;
    }

    // 转移Material
    int  materialIdx = 0;
    uint32_t MaterialNums = engine->materialMap.size();
    uint32_t texPixelNUms = 0;
    MaterialGPU MaterialsGPU[MaterialNums];  // MaterialGPU 数据容器
    std::unordered_map<std::shared_ptr<Material>, int> MaterialGPUMap;  // 临时记录 Material* - id 的映射结构
    for (const auto& material : engine->materialMap | std::views::values) {
        MaterialGPUMap[material] = materialIdx;
        MaterialsGPU[materialIdx].Ke = Vec3ToFloat3(material->Ke);  // 记录material的Ke
        // 记录Material的贴图像素的数量和贴图缓冲区偏移量
        const auto& KdMap = material->KdMap;
        if (!KdMap) {  // 记录纹理贴图Id，当没有贴图的时候赋值为-1
            MaterialsGPU[materialIdx].KdMapId = -1;
            MaterialsGPU[materialIdx].KdPixCount = 0;
            materialIdx++;
            continue;
        }
        MaterialsGPU[materialIdx].KdMapId = TextMap[KdMap];
        MaterialsGPU[materialIdx].KdPixCount = KdMap->uvImg->floatImg.size();
        MaterialsGPU[materialIdx].KdPixOffset = texPixelNUms;
        texPixelNUms+=MaterialsGPU[materialIdx].KdPixCount;  // 这里先统计贴图像素总数，后一个循环进行拷贝
        materialIdx++;
    }

    // VBO、EBO、SubMesh 数据容器准备
    uint32_t VBONums = 0;
    uint32_t EBONums = 0;
    uint32_t SubMeshNum = 0;
    uint32_t MeshIdx = 0;  // Mesh指针
    std::unordered_map<std::shared_ptr<Mesh>, size_t> MeshMapGPU;  // 临时记录 Mesh* - id 的映射结构
    for (const auto& instance : engine->tlas->instances) {
        const std::shared_ptr<Mesh> mesh = engine->blasList[instance.blasIdx]->mesh;
        if (MeshMapGPU.contains(mesh)) continue;  // 如果这个材质已经索引过了，则跳过
        MeshMapGPU[mesh] = MeshIdx++;  // 记录 Mesh* - id 映射
        VBONums += mesh->getVBONums();
        EBONums += mesh->getEBONums();
        SubMeshNum += mesh->getSubMeshNums();
    }
    SubMeshGPU SubMeshesGPU[SubMeshNum];
    VertexGPU VBO[VBONums];
    uint32_t EBO[EBONums];

    // 打包Mesh结构、VBO和EBO
    uint32_t meshNum = MeshMapGPU.size();
    MeshGPU MeshesGPU[meshNum];  // Mesh数据容器

    size_t VBOIdx = 0;  // VBO指针
    size_t EBOIdx = 0;  // EBO指针
    size_t SMIdx = 0;   // SubMesh指针

    // 遍历Mesh* - id Map获取对应Mesh
    for (const auto& [mesh, MeshId] : MeshMapGPU) {
        auto& meshGPU = MeshesGPU[MeshId];
        meshGPU.MeshEBOCount = mesh->getEBONums();
        meshGPU.MeshEBOffset = EBOIdx;

        meshGPU.MeshVBOCount = mesh->getVBONums();
        meshGPU.MeshVBOffset = VBOIdx;

        meshGPU.MeshSubCount = mesh->getSubMeshNums();
        meshGPU.MeshSubOffset = SMIdx;

        // 存储顶点数据
        for (const auto& vertex : mesh->VBO) {
            Vertex2GPU(vertex, VBO[VBOIdx++]);
        }
        // 存储Edge索引数据
        std::ranges::copy(mesh->EBO, EBO+EBOIdx);  // 似乎可以不用拷贝直接给一个指针
        EBOIdx += meshGPU.MeshEBOCount;  // 后续不能再使用EBOIdx

        // 存储SubMesh数据
        for (const auto &submesh : *mesh) {
            // 设置SubMesh的 绝对索引
            auto& submeshGPU = SubMeshesGPU[SMIdx++];
            submeshGPU.SubEBOffset = submesh.getOffset() + meshGPU.MeshEBOffset;  // EBO绝对索引
            submeshGPU.SubEBOCount = submesh.getIdxCount();
            submeshGPU.MaterialId = MaterialGPUMap[submesh.getMaterial()];  // 记录SubMesh的 Material id 代替指针
        }
    }
    ScenceDataGPU scenceData{
        MaterialsGPU,
        MaterialNums,
        VBO,
        VBONums,
        EBO,
        EBONums,
        MeshesGPU,
        meshNum,
        SubMeshesGPU,
        SubMeshNum,
        // texPixelsGPU,
        TextMap,
        texPixelNUms};
    // 上述代码打包了 cameraDataGPU VBO EBO MeshesGPU SubMeshesGPU MaterialsGPU texPixelsGPU -> cameraDataGPU scenceData

    // 打包TLAS数据
    TLASGPU tlasGPU{};
    tlasGPU.instanceCount = engine->tlas->instances.size();
    tlasGPU.nodeCount = engine->tlas->nodes.size();
    auto* instancesGPU = new InstanceGPU[tlasGPU.instanceCount];
    auto* nodesGPU = new BVHNodeGPU[tlasGPU.nodeCount];
    tlasGPU.instances = instancesGPU;
    tlasGPU.nodes = nodesGPU;

    const auto& TLASInsCPU = engine->tlas->instances.data();
    const auto& TLASNodesCPU = engine->tlas->nodes.data();

    std::memcpy(instancesGPU, TLASInsCPU, tlasGPU.instanceCount * sizeof(InstanceGPU));
    std::memcpy(nodesGPU, TLASNodesCPU, tlasGPU.nodeCount * sizeof(BVHNodeGPU));

    // 打包BLAS数据(BLAS是全局存储的Mesh的BVH数据)
    uint32_t BLASNums = engine->blasList.size();
    BLASGPU BlasGPU[BLASNums];
    uint32_t triNums = 0;
    uint32_t nodeNums = 0;
    for (uint32_t i = 0; i < BLASNums; ++i) {
        const auto& blasCPU = engine->blasList[i];
        BLASGPU& blasGPU = BlasGPU[i];
        blasGPU.nodeCount = blasCPU->nodes.size();
        blasGPU.triangleCount = blasCPU->triangles.size();
        blasGPU.MeshGPUId = MeshMapGPU[blasCPU->mesh];
        // 拷贝三角索引和节点数据
        blasGPU.triangleOffset = triNums;  // 这是绝对索引
        blasGPU.nodeOffset = nodeNums;

        blasGPU.triangleCount = blasCPU->triangles.size();
        blasGPU.nodeCount = blasCPU->nodes.size();

        triNums += blasGPU.triangleCount;
        nodeNums += blasGPU.nodeCount;
    }
    // 打包BLAS的Node和Tri数据
    uint32_t BlasTriGPU[triNums];  // 注意这里的三角形索引是相对索引不是绝对索引
    BVHNodeGPU BlasNodesGPU[nodeNums];
    for (size_t i = 0; i < BLASNums; ++i) {
        const auto& blasCPU = engine->blasList[i];
        BLASGPU& blasGPU = BlasGPU[i];
        // 这里拷贝的是相对索引，使用的时候需要加上Mesh索引偏移量
        memcpy(BlasTriGPU+blasGPU.triangleOffset, blasCPU->triangles.data(), blasGPU.triangleCount * sizeof(uint32_t));
        memcpy(BlasNodesGPU+blasGPU.nodeOffset, blasCPU->nodes.data(), blasGPU.nodeCount * sizeof(BVHNodeGPU));
    }

    BVHDataGPU bvh{BlasGPU, BlasTriGPU, triNums, BlasNodesGPU, nodeNums, BLASNums, tlasGPU};
    // 上面打包了 BlasGPU[] tlasGPU -> bvh
    Inter(SSP, maxDepth, cameraDataGPU, bvh, scenceData, finalResult);

    #else
    // CPU逻辑
    std::vector<std::future<std::vector<F2P>>> futures;
    constexpr size_t BLOCK_SIZE = 1024;
    futures.reserve((pixelCount + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (size_t i = 0; i < pixelCount; i += BLOCK_SIZE) {
        size_t start = i;
        size_t end   = std::min(i + BLOCK_SIZE, pixelCount);

        futures.emplace_back(engine->pool.addTask(
            [&, start, end, width, height, Asp, scale, SSP, maxDepth] {
                std::vector<F2P> localResult;
                localResult.reserve(end - start);

                for (size_t idx = start; idx < end; ++idx) {
                    const size_t y = idx / width;
                    const size_t x = idx % width;
                    const float ndcX = 2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(width) - 1.0f;
                    const float ndcY = 1.0f - (2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(height));
                    const float camX = ndcX * Asp * scale;
                    const float camY = ndcY * scale;
                    constexpr float camZ = -1.0f;
                    const Vec4 rayDirCamera{camX, camY, camZ, 0.0f};
                    const Vec4 rayDirWorld = normalize(cameraRot * rayDirCamera);
                    const Ray ray(cameraPos, rayDirWorld);
                    Vec3 radiance(0.0f);
                    // Path Tracing
                    for (int s = 0; s < SSP; ++s) {
                        Ray currentRay = ray;
                        Vec3 throughput(1.0f);
                        for (int depth = 0; depth < maxDepth; ++depth) {
                            const auto hitInfo = engine->GetClosestHit(currentRay);
                            if (!hitInfo) break;
                            const auto material = hitInfo->mat;
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
                    radiance /= SSP;
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
    for (auto& part : allParts) {
        finalResult.insert(
            finalResult.end(),
            std::make_move_iterator(part.begin()),
            std::make_move_iterator(part.end())
        );
    }
    #endif
    // 写入缓冲区
    WriteBuffer(finalResult);
}

// 单个像素写入帧缓冲区：将一个F2P数据写入渲染结果缓冲区
void Graphic::WriteBuffer(const F2P& f2p) const {
    if (!f2p.alive) return;
    const size_t width = engine->width;
    engine->tmpBufferF[f2p.x + f2p.y * width] = f2p.Albedo;
}

// 批量写入帧缓冲区：将F2P数据写入渲染结果缓冲区（支持并行优化）
void Graphic::WriteBuffer(const std::vector<F2P>& f2pVec) const {
    const size_t count = f2pVec.size();
    if (count == 0) return;

    // 如果像素数量很少（1e4），直接主线程串行写完
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
            [&f2pVec, &buffer, width, start, end] {
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

// 写入G-Buffer：将片段的法线和世界坐标信息写入延迟渲染缓冲区
void Graphic::WriteGBuffer(const Fragment& frag) const {
    if (!frag.alive) return;
    const size_t width = engine->width;
    auto& gData = engine->gBuffer->Gdata;
    const auto idx = frag.x + frag.y * width;
    gData[idx].normal = frag.normal;
    gData[idx].worldPosi = frag.worldPosi;
}
