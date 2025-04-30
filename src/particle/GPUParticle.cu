#include "Particle.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <curand_kernel.h>
#include <stdio.h>

GPUParticles::GPUParticles(const int &particleNum) : N(particleNum)
{   
    size_t sizedoubles {N * sizeof(double)};   
    size_t sizedouble3s {N * sizeof(double3)};
    size_t sizeints {N * sizeof(int)};
    cudaMalloc((void**)&d_mass, sizedoubles);
    cudaMalloc((void**)&d_pos, sizedouble3s);
    cudaMalloc((void**)&d_vel, sizedouble3s);
    cudaMalloc((void**)&global_id, sizeints);
	cudaMalloc((void**)&global_id_sortted, sizeints);
    cudaMalloc((void**)&cell_id, sizeints);
    cudaMalloc((void**)&local_id, sizeints);

}

GPUParticles::~GPUParticles()
{
    cudaFree(d_mass);
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(global_id);
	cudaFree(global_id_sortted);
    cudaFree(local_id);
    cudaFree(cell_id);
}

void GPUParticles::UploadFromHost(const double* h_mass,
    const double3* h_pos,
    const double3* h_vel, 
    const int* h_global_id, const int* h_local_id, const int* h_cell_id)
{
    size_t sizedoubles {N * sizeof(double)};   
    size_t sizedouble3s {N * sizeof(double3)};
    size_t sizeints {N * sizeof(int)};
    cudaMemcpy(d_mass, h_mass, sizedoubles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos, sizedouble3s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, sizedouble3s, cudaMemcpyHostToDevice);

    cudaMemcpy(global_id, h_global_id, sizeints, cudaMemcpyHostToDevice);
    cudaMemcpy(local_id, h_local_id, sizeints, cudaMemcpyHostToDevice);
    cudaMemcpy(cell_id, h_cell_id, sizeints, cudaMemcpyHostToDevice);
}

void GPUParticles::Move(const double &dt, const double &blockSize, const Boundary* d_boundaries)
{
    int numBlocks = (N + blockSize - 1) / blockSize;
    GPUParticleKernels::moveParticles<<<numBlocks, blockSize>>>(d_pos,
                                          d_vel,
                                          N, dt, d_boundaries);
    cudaDeviceSynchronize();
}

void GPUParticles::Sort(const int* d_particleStartIndex)
{	
	int blockSize = 128;
	int numBlocks = (N + blockSize - 1) / blockSize;
	GPUParticleKernels::sortParticles<<<numBlocks, blockSize>>>(cell_id, local_id, global_id, global_id_sortted, d_particleStartIndex, N);
}

__global__ void GPUParticleKernels::moveParticles(double3* pos,
                               double3* vel,
                               int N, double dt, const Boundary* d_boundaries) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) 
        return;
    
    // 读取到register memory
    auto local_pos = pos[i];
    auto local_vel = vel[i];
    double d_Vstd = 330;
    curandState localState;
    curand_init(clock64() + 1234 * threadIdx.x + blockIdx.x, 0, 0, &localState);

    local_pos.x += local_vel.x * dt;
    local_pos.y += local_vel.y * dt;
    local_pos.z += local_vel.z * dt;
    /*============边界条件=============*/
    if(local_pos.x < 0){
        auto boundary = d_boundaries[0];
        // GPUBoundary::WallBoundary::apply(local_pos, local_vel, boundary.point, boundary.normal);
        // GPUBoundary::PeriodicBoundary::apply(local_pos, boundary.point, boundary.normal);
        local_pos.x = fmod(local_pos.x, L1) + L1;
    }
    if(local_pos.x > L1){
        auto boundary = d_boundaries[1];
        // GPUBoundary::WallBoundary::apply(local_pos, local_vel, boundary.point, boundary.normal);
        // GPUBoundary::PeriodicBoundary::apply(local_pos, boundary.point, boundary.normal);
        local_pos.x = fmod(local_pos.x, L1);
    }
    if(local_pos.y < 0){
        // auto boundary = d_boundaries[2];
        // GPUBoundary::WallBoundary::apply(local_pos, local_vel, boundary.point, boundary.normal);
        auto dt_ac = (local_pos.y) / local_vel.y;
        auto rand1 {curand_uniform(&localState)};
        auto a1 = sqrt(-log(rand1)) * d_Vstd;
        auto rand2 {curand_uniform(&localState)};
        auto a2 = 2 * M_PI * rand2;
        local_vel.x = sin(a2) * a1;
        local_vel.z = cos(a2) * a1;
        
        double rand3 = curand_uniform(&localState);
        local_vel.y = fabs(sqrt(-log(rand3)) * d_Vstd);
        local_pos.y = fabs(local_vel.y * dt_ac);
    }
    if (local_pos.y > L2) {
        // auto boundary = d_boundaries[3];
        // GPUBoundary::WallBoundary::apply(local_pos, local_vel, boundary.point, boundary.normal);
        // local_vel.x = 300;
        double dt_ac = (local_pos.y - L2) / local_vel.y; // old vy > 0
    
        // 采样一个新的速度（热壁反射 + 驱动盖速度）
        double rand1 = curand_uniform(&localState);
        double a1 = sqrt(-log(rand1)) * d_Vstd;  // 热壁麦克斯韦速度采样
        double rand2 = curand_uniform(&localState);
        double a2 = 2.0 * M_PI * rand2;
    
        local_vel.x = a1 * cos(a2) + 300.0;  // 加上顶盖速度
        local_vel.z = a1 * sin(a2);
    
        double rand3 = curand_uniform(&localState);
        local_vel.y = -sqrt(-log(rand3)) * d_Vstd;  // 反向的vy
        // printf("New velocity vy: %f\n", local_vel.y);
        // 回退新位置（反弹之后的位移）
        local_pos.y = L2 - fabs(local_vel.y * dt_ac);  // new vy 是负数
    }
    
    if(local_pos.z < 0){
        local_pos.z = fmod(local_pos.z, L3) + L3;
    }

    if(local_pos.z > L3){
        local_pos.z = fmod(local_pos.z, L3);
    }

    // 写回global memory
    pos[i] = local_pos;
    vel[i] = local_vel;
}

__global__ void GPUParticleKernels::sortParticles(const int *cell_id, const int *local_id, const int *global_id, int *global_id_sortted, const int *d_particleStartIndex, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) 
        return;
	
	int sorted_global_id = d_particleStartIndex[cell_id[i]] + local_id[i];
	global_id_sortted[sorted_global_id] = global_id[i];
}
