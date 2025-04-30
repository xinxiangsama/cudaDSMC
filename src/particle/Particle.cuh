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
    void Sort(const int* d_particleStartIndex);
// protected:
    double* d_mass;
    double3* d_pos;
    double3* d_vel;
    int* global_id;
    int* global_id_sortted;
    int* cell_id;
    int* local_id;
    int N;
};


namespace GPUParticleKernels {
    using Boundary = GPUBoundary::Boundary;

    __global__ void moveParticles(double3* pos,
                                  double3* vel,
                                  int N, double dt, const Boundary* d_boundaries,
                                  const int* d_ifCut, const GPUSegment* d_Segments, const int* CellID);
    
    __global__ void sortParticles(const int* cell_id, const int* local_id, const int* global_id, int* global_id_sortted, const int* d_particleStartIndex, int N);
}