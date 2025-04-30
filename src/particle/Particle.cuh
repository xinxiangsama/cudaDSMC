#pragma once
#include <cuda_runtime.h>
#include "../boundary/Boundary.cuh"

// 包含所有GPU端粒子的信息和方法
class GPUParticles{
    using Boundary = GPUBoundary::Boundary;
public:
    GPUParticles() = default;
    GPUParticles(const int& particleNum);
    ~GPUParticles();
    void UploadFromHost(const double* h_mass,
        const double* h_pos_x, const double* h_pos_y, const double* h_pos_z,
        const double* h_vel_x, const double* h_vel_y, const double* h_vel_z, const int* h_global_id, const int* h_local_id, const int* h_cell_id);
    
    void Move(const double& dt, const double& blockSize, const Boundary* d_boundaries);
    void Sort(const int* d_particleStartIndex);
// protected:
    double* d_mass;
    double* d_pos_x;
    double* d_pos_y;
    double* d_pos_z;
    double* d_vel_x;
    double* d_vel_y;
    double* d_vel_z;
    int* global_id;
    int* global_id_sortted;
    int* cell_id;
    int* local_id;
    int N;
};


namespace GPUParticleKernels {
    using Boundary = GPUBoundary::Boundary;

    __global__ void moveParticles(double* pos_x, double* pos_y, double* pos_z,
                                  double* vel_x, double* vel_y, double* vel_z,
                                  int N, double dt, const Boundary* d_boundaries);
    
    __global__ void sortParticles(const int* cell_id, const int* local_id, const int* global_id, int* global_id_sortted, const int* d_particleStartIndex, int N);
}