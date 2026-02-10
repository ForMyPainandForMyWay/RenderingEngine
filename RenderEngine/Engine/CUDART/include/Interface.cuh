//
// Created by yyd on 2026/2/2.
//

#ifndef RENDERINGENGINE_INTERFACE_CUH
#define RENDERINGENGINE_INTERFACE_CUH
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

struct F2P;
struct ScenceDataGPU;
struct BVHDataGPU;
struct CameraDataGPU;


bool __host__ checkErrorFail(cudaError_t error);
void __host__ Inter(
    std::uint8_t SSP,
    uint8_t maxDepth,
    const CameraDataGPU& CmDataCPU, const BVHDataGPU& bvh, const ScenceDataGPU& scenceData, std::vector<F2P> &result);

#endif //RENDERINGENGINE_INTERFACE_CUH