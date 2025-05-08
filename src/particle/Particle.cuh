#pragma once
#include <cuda_runtime.h>
#include "../boundary/Boundary.cuh"
#include "../constant/Paramconstants.cuh"
#include "object/GPUSegment.cuh"
// 包含所有GPU端粒子的信息和方法
class GPUParticles{
    using Boundary = GPUBoundary::Boundary;
public:
    GPUParticles() = default;
    GPUParticles(const int& particleNum);
    ~GPUParticles();
    void UploadFromHost(const double* h_mass,
        const double3* h_pos,
        const double3* h_vel, 
        const int* h_global_id, const int* h_local_id, const int* h_cell_id);
    
    void Move(const double& dt, const double& blockSize, 
            const Boundary* d_boundaries,
            const int* d_ifCut, const GPUSegment* d_Segments);
    void DeleteInvalid(int* d_valid);
    void Sort(const int* d_particleStartIndex);
    void Injet();
    void ResizeStorage(const int& newCapacity);
// protected:
    double* d_mass;
    double3* d_pos;
    double3* d_vel;
    int* global_id;
    int* global_id_sortted;
    int* cell_id;
    int* local_id;
    int* d_injectedCounter;
    int N;
    int m_Capacity;
};


namespace GPUParticleKernels {
    using Boundary = GPUBoundary::Boundary;

    __global__ void moveParticles(double3* pos,
                                  double3* vel,
                                  int N, double dt, const Boundary* d_boundaries,
                                  const int* d_ifCut, const GPUSegment* d_Segments, const int* CellID, int* d_valid);
    
    __global__ void sortParticles(const int* cell_id, const int* local_id, const int* global_id, int* global_id_sortted, const int* d_particleStartIndex, int N);

    __global__ void InjectParticles(
        double3* d_pos,
        double3* d_vel,
        int*     d_globalID,
        int N,                      // 已有的粒子数
        int maxInject,              // 注入的粒子数
        int* d_injectedCounter      // 原子变量，记录已注入粒子数（决定 global_id）
    ) ;
}

namespace GPURandomKernels{
    __device__ double3 MaxwellDistribution(const double& Vstd, curandState& localState);
}