//
// Created by yyd on 2026/2/2.
//
#include <iostream>

#include "Interface.cuh"
#include "DATAPackegGPU.cuh"
#include "F2P.hpp"
#include "Mesh.cuh"
#include "Mesh.hpp"
#include "PathTracing.cuh"

// 检查是否cuda失败
bool __host__ checkErrorFail(const cudaError_t error) {
    if (error == cudaSuccess) return false;
    if (error == cudaErrorMemoryAllocation) {
        std::cerr << "cudaErrorMemoryAllocation: " << cudaGetErrorString(error) << std::endl;
    } else if (error == cudaErrorInvalidValue) {
        std::cerr << "cudaErrorInvalidValue: " << cudaGetErrorString(error) << std::endl;
    } else {
        std::cerr << "cudaError: " << cudaGetErrorString(error) << std::endl;
    }
    return true;
}

void __host__ Inter(
    const std::uint8_t SSP,
    const uint8_t maxDepth,
    const CameraDataGPU& CmDataCPU, const BVHDataGPU& bvh, const ScenceDataGPU& scenceData, std::vector<F2P>& result) {
    // 这里拿到了打包好的数据
    // 先传送 相机数据 BVH 到 GPU常量内存
    cudaError_t error = cudaMemcpyToSymbol(CmDataGPu, &CmDataCPU, sizeof(CameraDataGPU));
    if (checkErrorFail(error)) return;
    error = cudaMemcpyToSymbol(TlasNodesGPU, bvh.tlas.nodes, sizeof(BVHNodeGPU) * bvh.tlas.nodeCount);
    if (checkErrorFail(error)) return;
    error = cudaMemcpyToSymbol(TlasInstanceGPU, bvh.tlas.instances, sizeof(InstanceGPU) * bvh.tlas.instanceCount);
    if (checkErrorFail(error)) return;
    error = cudaMemcpyToSymbol(tlasNodeNums, &bvh.tlas.nodeCount, sizeof(size_t));
    if (checkErrorFail(error)) return;
    error = cudaMemcpyToSymbol(tlasInstanceNums, &bvh.tlas.instanceCount, sizeof(size_t));
    if (checkErrorFail(error)) return;


    // BLAS大小不足以存储在GPU常量内存中，需要在GPU内存中申请
    BLASGPU* h_blasGPU;
    error = cudaMalloc(&h_blasGPU, sizeof(BLASGPU) * bvh.blasCount);
    if (checkErrorFail(error)) return;
    error = cudaMemcpy(h_blasGPU, bvh.blas, sizeof(BLASGPU) * bvh.blasCount, cudaMemcpyHostToDevice);
    if (checkErrorFail(error)) return;

    // 三角形的相对索引，使用时需要加上blas的offset
    uint32_t* h_BlasTriGPU;
    BVHNodeGPU* h_BlasNodesGPU;

    error = cudaMalloc(&h_BlasTriGPU, sizeof(uint32_t) * bvh.triNums);
    if (checkErrorFail(error)) return;
    error = cudaMemcpy(h_BlasTriGPU, bvh.BlasTriGPU, sizeof(uint32_t) * bvh.triNums, cudaMemcpyHostToDevice);
    if (checkErrorFail(error)) return;
    error = cudaMalloc(&h_BlasNodesGPU, sizeof(BVHNodeGPU) * bvh.nodeNums);
    if (checkErrorFail(error)) return;
    error = cudaMemcpy(h_BlasNodesGPU, bvh.BlasNodesGPU, sizeof(BVHNodeGPU) * bvh.nodeNums, cudaMemcpyHostToDevice);
    if (checkErrorFail(error)) return;


    // 拷贝Mesh相关的数据
    VertexGPU* h_vboGPU;
    uint32_t* h_eboGPU;
    MeshGPU* h_meshesGPU;
    SubMeshGPU* h_subMeshesGPU;
    MaterialGPU* h_materialsGPU;


    error = cudaMalloc(&h_vboGPU, sizeof(VertexGPU) * scenceData.vboCount);
    if (checkErrorFail(error)) return;
    error = cudaMalloc(&h_eboGPU, sizeof(uint32_t) * scenceData.eboCount);
    if (checkErrorFail(error)) return;
    error = cudaMalloc(&h_meshesGPU, sizeof(MeshGPU) * scenceData.meshCount);
    if (checkErrorFail(error)) return;
    error = cudaMalloc(&h_subMeshesGPU, sizeof(SubMeshGPU) * scenceData.subMeshCount);
    if (checkErrorFail(error)) return;
    error = cudaMalloc(&h_materialsGPU, sizeof(MaterialGPU) * scenceData.MaterialCount);

    error = cudaMemcpy(h_vboGPU, scenceData.vboGPU, sizeof(VertexGPU) * scenceData.vboCount, cudaMemcpyHostToDevice);
    if (checkErrorFail(error)) return;
    error = cudaMemcpy(h_eboGPU, scenceData.eboGPU, sizeof(uint32_t) * scenceData.eboCount, cudaMemcpyHostToDevice);
    if (checkErrorFail(error)) return;
    error = cudaMemcpy(h_meshesGPU, scenceData.MeshesGPU, sizeof(MeshGPU) * scenceData.meshCount, cudaMemcpyHostToDevice);
    if (checkErrorFail(error)) return;
    error = cudaMemcpy(h_subMeshesGPU, scenceData.SubMeshesGPU, sizeof(SubMeshGPU) * scenceData.subMeshCount, cudaMemcpyHostToDevice);
    if (checkErrorFail(error)) return;
    error = cudaMemcpy(h_materialsGPU, scenceData.MatGPU, sizeof(MaterialGPU) * scenceData.MaterialCount, cudaMemcpyHostToDevice);

    // 材质数据
    error = cudaMalloc(&h_materialsGPU, sizeof(MaterialGPU) * scenceData.MaterialCount);
    if (checkErrorFail(error)) return;
    error = cudaMemcpy(h_materialsGPU, scenceData.MatGPU, sizeof(MaterialGPU) * scenceData.MaterialCount, cudaMemcpyHostToDevice);
    if (checkErrorFail(error)) return;

    // 符号注入
    cudaMemcpyToSymbol(vboGPU, &h_vboGPU, sizeof(VertexGPU*));
    cudaMemcpyToSymbol(eboGPU, &h_eboGPU, sizeof(uint32_t*));
    cudaMemcpyToSymbol(meshesGPU, &h_meshesGPU, sizeof(MeshGPU*));
    cudaMemcpyToSymbol(subMeshesGPU, &h_subMeshesGPU, sizeof(SubMeshGPU*));
    cudaMemcpyToSymbol(materialsGPU, &h_materialsGPU, sizeof(MaterialGPU*));
    cudaMemcpyToSymbol(blasGPU, &h_blasGPU, sizeof(BLASGPU*));
    cudaMemcpyToSymbol(BlasTriGPU, &h_BlasTriGPU, sizeof(uint32_t*));
    cudaMemcpyToSymbol(BlasNodesGPU, &h_BlasNodesGPU, sizeof(BVHNodeGPU*));

    // 纹理数据
    // 构建资源描述符（指定底层存储）
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>(); // RGBA float
    auto texObjsHost = new cudaTextureObject_t[scenceData.TextMap.size()];
    for (const auto& [tex, texId] : scenceData.TextMap) {
        cudaArray_t cuArray;
        error = cudaMallocArray(&cuArray, &channelDesc, tex->width, tex->height);
        if (checkErrorFail(error)) return;
        error = cudaMemcpy2DToArray(
                cuArray,
                0,0,
                tex->uvImg->floatImg.data(),
                tex->width * sizeof(float4),
                tex->width * sizeof(float4),
                tex->height,
                cudaMemcpyHostToDevice
                );
        if (checkErrorFail(error)) return;
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;  // 直接赋值 cudaArray_t
        // 纹理描述符
        cudaTextureDesc texDec{};
        texDec.addressMode[0] = cudaAddressModeWrap;
        texDec.filterMode = cudaFilterModeLinear;  // 线性插值
        texDec.readMode = cudaReadModeElementType;  // 直接返回元素值
        texDec.normalizedCoords = 1;  // 0整数坐标 1浮点坐标
        cudaTextureObject_t texObj;
        error = cudaCreateTextureObject(&texObj, &resDesc, &texDec, nullptr);
        if (checkErrorFail(error)) return;
        texObjsHost[texId] = texObj;  // 存储纹理对象
    }
    // 传送纹理对象
    error = cudaMemcpyToSymbol(texObjsGPU, texObjsHost, sizeof(cudaTextureObject_t) * scenceData.TextMap.size());
    if (checkErrorFail(error)) return;

    // 主机端启动
    const int totalPixels = static_cast<int>(CmDataCPU.width * CmDataCPU.height);
    constexpr int threadsPerBlock = 256; //  ≤1024
    const int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock; // 向上取整
    // 结果缓冲区
    F2PGPU* resultGPU;
    cudaMalloc(&resultGPU, sizeof(F2PGPU) * totalPixels);
    // 启动核函数
    pathTracing<<<blocksPerGrid, threadsPerBlock>>>(resultGPU, SSP, maxDepth);

    result.resize(totalPixels);
    error = cudaMemcpy(
    result.data(),          // 目标：vector底层缓冲区
    resultGPU,                // 源：设备指针
    totalPixels * sizeof(F2P),  // 字节数
    cudaMemcpyDeviceToHost);
    checkErrorFail(error);
}
