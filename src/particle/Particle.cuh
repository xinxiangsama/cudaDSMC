#pragma once
#include <cuda_runtime.h>

// 包含所有GPU端粒子的信息和方法
class GPUParticles{
public:
    GPUParticles() = default;
    GPUParticles(const int& particleNum);
    ~GPUParticles();
    void UploadFromHost(const double* h_mass,
        const double* h_pos_x, const double* h_pos_y, const double* h_pos_z,
        const double* h_vel_x, const double* h_vel_y, const double* h_vel_z, const int* h_global_id, const int* h_local_id, const int* h_cell_id);
    
    void Move(const double& dt, const double& blockSize);
    void Sort();
// protected:
    double* d_mass;
    double* d_pos_x;
    double* d_pos_y;
    double* d_pos_z;
    double* d_vel_x;
    double* d_vel_y;
    double* d_vel_z;
    int* global_id;
    int* cell_id;
    int* local_id;
    int N;
};


namespace GPUParticleKernels {

    __global__ void moveParticles(double* pos_x, double* pos_y, double* pos_z,
                                  const double* vel_x, const double* vel_y, const double* vel_z,
                                  int N, double dt);
    
}