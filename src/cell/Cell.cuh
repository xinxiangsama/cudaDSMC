#pragma once
#include <cuda_runtime.h>
#include <vector>


class GPUCells
{
public:
    GPUCells(const int& N);
    ~GPUCells();
    void UploadFromHost(const int* h_particleNum, const int* h_particleStartIndex, const double& h_Unidx, const double& h_Unidy, const double& h_Unidz);

    void DownloadToHost(const double* h_Temperature, const double* h_Rho, const double* h_Pressure, const double3* h_Velocity);
    void CalparticleNum(const double* __restrict__ d_pos_x,
        const double* __restrict__ d_pos_y,
        const double* __restrict__ d_pos_z,
        int* __restrict__ d_localID,
        int* __restrict__ d_cellID,
        int particleNum,
        int nx, int ny, int nz, int blockSize);
    
    void CalparticleStartIndex();
    void Collision(double* d_vel_x, double* d_vel_y, double* d_vel_z);
    void Sample(const double* d_vel_x, const double* d_vel_y, const double* d_vel_z, int totalParticles);
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

    __global__ void collisionInCells(
        double* d_vel_x, double* d_vel_y, double* d_vel_z,
        const int* __restrict__ d_particleStartIndex,
        const int* __restrict__ d_particleNum,
        int totalCells
    );
    
    __global__ void samplingInCells(
        const double* __restrict__ d_vel_x,
        const double* __restrict__ d_vel_y,
        const double* __restrict__ d_vel_z,
        const int* __restrict__ d_particleStartIndex,
        const int* __restrict__ d_particleNum,
        double* __restrict__ d_Temperature,
        double* __restrict__ d_Rho,
        double3* __restrict__ d_Velocity,
        double* __restrict__ d_Pressure,
        int totalCells,
        int totalParticles
    );
}