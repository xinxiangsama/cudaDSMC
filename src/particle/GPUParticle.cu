#include "Particle.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>


extern __constant__ double d_Unidx;
extern __constant__ double d_Unidy;
extern __constant__ double d_Unidz;

GPUParticles::GPUParticles(const int &particleNum) : N(particleNum)
{   
    size_t sizedoubles {N * sizeof(double)};   
    size_t sizeints {N * sizeof(int)};
    cudaMalloc((void**)&d_mass, sizedoubles);
    cudaMalloc((void**)&d_pos_x, sizedoubles);
    cudaMalloc((void**)&d_pos_y, sizedoubles);
    cudaMalloc((void**)&d_pos_z, sizedoubles);
    cudaMalloc((void**)&d_vel_x, sizedoubles);
    cudaMalloc((void**)&d_vel_y, sizedoubles);
    cudaMalloc((void**)&d_vel_z, sizedoubles);  
    cudaMalloc((void**)&global_id, sizeints);
    cudaMalloc((void**)&cell_id, sizeints);
    cudaMalloc((void**)&local_id, sizeints);

}

GPUParticles::~GPUParticles()
{
    cudaFree(d_mass);
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_pos_z);
    cudaFree(d_vel_x);
    cudaFree(d_vel_y);
    cudaFree(d_vel_z);
    cudaFree(global_id);
    cudaFree(local_id);
    cudaFree(cell_id);
}

void GPUParticles::UploadFromHost(const double* h_mass,
    const double* h_pos_x, const double* h_pos_y, const double* h_pos_z,
    const double* h_vel_x, const double* h_vel_y, const double* h_vel_z, const int* h_global_id, const int* h_local_id, const int* h_cell_id)
{
    size_t sizedoubles {N * sizeof(double)};
    size_t sizeints {N * sizeof(int)};
    cudaMemcpy(d_mass, h_mass, sizedoubles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_x, h_pos_x, sizedoubles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, h_pos_y, sizedoubles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_z, h_pos_z, sizedoubles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_x, h_vel_x, sizedoubles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_y, h_vel_y, sizedoubles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_z, h_vel_z, sizedoubles, cudaMemcpyHostToDevice);

    cudaMemcpy(global_id, h_global_id, sizeints, cudaMemcpyHostToDevice);
    cudaMemcpy(local_id, h_local_id, sizeints, cudaMemcpyHostToDevice);
    cudaMemcpy(cell_id, h_cell_id, sizeints, cudaMemcpyHostToDevice);
}

void GPUParticles::Move(const double &dt, const double &blockSize)
{
    int numBlocks = (N + blockSize - 1) / blockSize;
    GPUParticleKernels::moveParticles<<<numBlocks, blockSize>>>(d_pos_x, d_pos_y, d_pos_z,
                                          d_vel_x, d_vel_y, d_vel_z,
                                          N, dt);
}

void GPUParticles::Sort()
{
    // 把d_cellID作为key，其他属性作为value，按cellID升序排列
    thrust::device_ptr<int> d_cellID_ptr = thrust::device_pointer_cast(cell_id);
    thrust::device_ptr<double> d_pos_x_ptr = thrust::device_pointer_cast(d_pos_x);
    thrust::device_ptr<double> d_pos_y_ptr = thrust::device_pointer_cast(d_pos_y);
    thrust::device_ptr<double> d_pos_z_ptr = thrust::device_pointer_cast(d_pos_z);
    thrust::device_ptr<double> d_vel_x_ptr = thrust::device_pointer_cast(d_vel_x);
    thrust::device_ptr<double> d_vel_y_ptr = thrust::device_pointer_cast(d_vel_y);
    thrust::device_ptr<double> d_vel_z_ptr = thrust::device_pointer_cast(d_vel_z);
    thrust::device_ptr<double> d_mass_ptr = thrust::device_pointer_cast(d_mass);

    // 按cellID升序排列粒子
    thrust::sort_by_key(d_cellID_ptr, d_cellID_ptr + N, d_pos_x_ptr);
    thrust::sort_by_key(d_cellID_ptr, d_cellID_ptr + N, d_pos_y_ptr);
    thrust::sort_by_key(d_cellID_ptr, d_cellID_ptr + N, d_pos_z_ptr);
    thrust::sort_by_key(d_cellID_ptr, d_cellID_ptr + N, d_vel_x_ptr);
    thrust::sort_by_key(d_cellID_ptr, d_cellID_ptr + N, d_vel_y_ptr);
    thrust::sort_by_key(d_cellID_ptr, d_cellID_ptr + N, d_vel_z_ptr);
    thrust::sort_by_key(d_cellID_ptr, d_cellID_ptr + N, d_mass_ptr);

}

__global__ void GPUParticleKernels::moveParticles(double* pos_x, double* pos_y, double* pos_z,
                                const double* vel_x, const double* vel_y, const double* vel_z,
                                int N, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        pos_x[i] += vel_x[i] * dt;
        pos_y[i] += vel_y[i] * dt;
        pos_z[i] += vel_z[i] * dt;

        // 应用边界条件
    }
}
    