#pragma once
#include <cuda_runtime.h>
#include <vector>


class GPUCells
{
public:
    GPUCells(const int& N);
    ~GPUCells();
    void UploadFromHost(const int* h_particleNum, const int* h_particleStartIndex, const double& h_Unidx, const double& h_Unidy, const double& h_Unidz);

    void CalparticleNum(const double* __restrict__ d_pos_x,
        const double* __restrict__ d_pos_y,
        const double* __restrict__ d_pos_z,
        int* __restrict__ d_localID,
        int* __restrict__ d_cellID,
        int particleNum,
        int nx, int ny, int nz, int blockSize);
    
    void CalparticleStartIndex();
// protected:
    int* d_particleNum {};
    int* d_particleStartIndex {};
    double* d_Temperature {};
    double* d_Rho {};
    double* d_Pressure {};
    double3* d_Velocity {};

    size_t m_cellNum;
};

namespace GPUCellKernels{
    // 计算每个cell中的粒子数，以及每个粒子在cell中的local id
    __global__ void CalparticleNumAndLocalID(
        const double* __restrict__ d_pos_x,
        const double* __restrict__ d_pos_y,
        const double* __restrict__ d_pos_z,
        int* __restrict__ d_particleNum,
        int* __restrict__ d_localID,
        int* __restrict__ d_cellID,
        int particleNum,
        int nx, int ny, int nz
    );
}