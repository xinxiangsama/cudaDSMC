#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "object/GPUSegment.cuh"


class GPUCells
{
public:
    GPUCells(const int& N);
    ~GPUCells();
    void UploadFromHost(const int* h_particleNum, const int* h_particleStartIndex, const double& h_Unidx, const double& h_Unidy, const double& h_Unidz, const int* h_ifCut, const GPUSegment* h_Segments);
    void DownloadToHost(const double* h_Temperature, const double* h_Rho, const double* h_Pressure, const double3* h_Velocity);
    void CalparticleNum(const double3* __restrict__ d_pos,
        int* __restrict__ d_localID,
        int* __restrict__ d_cellID,
        int particleNum,
        int nx, int ny, int nz, int blockSize);
    
    void CalparticleStartIndex();
    void Collision(double3* d_vel, int* global_id_sortted);
    void Sample(const double3* d_vel, int totalParticles, int* global_id_sortted);
// protected:
    int* d_particleNum {};
    int* d_particleStartIndex {};
    int* d_collisionCount;
    double* d_Temperature {};
    double* d_Rho {};
    double* d_Pressure {};
    double3* d_Velocity {};
    int* d_ifCut {};
    GPUSegment* d_Segments {};

    size_t m_cellNum;
};

namespace GPUCellKernels{
    // 计算每个cell中的粒子数，以及每个粒子在cell中的local id
    __global__ void CalparticleNumAndLocalID(
        const double3* __restrict__ d_pos,
        int* __restrict__ d_particleNum,
        int* __restrict__ d_localID,
        int* __restrict__ d_cellID,
        int particleNum,
        int nx, int ny, int nz
    );

    __global__ void collisionInCells(
        double3* d_vel,
        const int* __restrict__ d_particleStartIndex,
        const int* __restrict__ d_particleNum,
        int totalCells,
        int* global_id_sortted,
        int* d_collisionCount
    );
    
    __global__ void samplingInCells(
        const double3* __restrict__ d_vel,
        const int* __restrict__ d_particleStartIndex,
        const int* __restrict__ d_particleNum,
        double* __restrict__ d_Temperature,
        double* __restrict__ d_Rho,
        double3* __restrict__ d_Velocity,
        double* __restrict__ d_Pressure,
        int totalCells,
        int totalParticles,
        int* global_id_sortted
    );
}