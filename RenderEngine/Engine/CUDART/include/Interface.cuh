//
// Created by yyd on 2026/2/2.
//

#ifndef RENDERINGENGINE_INTERFACE_CUH
#define RENDERINGENGINE_INTERFACE_CUH
#include <vector>
#include <cuda_runtime.h>

struct F2P;
struct ScenceDataGPU;
struct BVHDataGPU;
struct CameraDataGPU;


bool __host__ checkErrorFail(cudaError_t error);
void __host__ Inter(const CameraDataGPU& CmDataCPU, const BVHDataGPU& bvh, const ScenceDataGPU& scenceData, std::vector<F2P> &result);

#endif //RENDERINGENGINE_INTERFACE_CUH