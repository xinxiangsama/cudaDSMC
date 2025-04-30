#include "Cell.cuh"
#include "Param.h"
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <curand_kernel.h>

extern double Vstd;
extern double Vmax;
extern double VHS_coe;
__device__ __constant__ double d_Unidx;
__device__ __constant__ double d_Unidy;
__device__ __constant__ double d_Unidz;
__device__ __constant__ double d_L1;
__device__ __constant__ double d_L2;
__device__ __constant__ double d_L3;
__device__ __constant__ double d_diam;
__device__ __constant__ double d_boltz;
__device__ __constant__ double d_Vtl;
__device__ __constant__ double d_VHS;
__device__ __constant__ double d_Vmax;
__device__ __constant__ double d_Vstd;
__device__ __constant__ double d_tau;
__device__ __constant__ size_t d_Fn;

GPUCells::GPUCells(const int& N) : m_cellNum(N)
{
    size_t sizei = N * sizeof(int);
    size_t sized = N * sizeof(double);
    size_t size3d = N * sizeof(double3);
    size_t sizeb = N * sizeof(bool);

    cudaMalloc((void**)&d_particleNum, sizei);
    cudaMalloc((void**)&d_particleStartIndex, sizei);
    cudaMalloc((void**)&d_collisionCount, sizei);
    cudaMalloc((void**)&d_Temperature, sized);
    cudaMalloc((void**)&d_Rho, sized);
    cudaMalloc((void**)&d_Pressure, sized);
    cudaMalloc((void**)&d_Velocity, size3d);
    cudaMalloc((void**)&d_ifCut, sizei);
    cudaMalloc((void**)&d_Segments, N * sizeof(GPUSegment));

    cudaMemcpyToSymbol(d_L1, &L1, sizeof(double));
    cudaMemcpyToSymbol(d_L2, &L2, sizeof(double));
    cudaMemcpyToSymbol(d_L2, &L2, sizeof(double));
    cudaMemcpyToSymbol(d_L3, &L3, sizeof(double));
    cudaMemcpyToSymbol(d_diam, &diam, sizeof(double));
    cudaMemcpyToSymbol(d_boltz, &boltz, sizeof(double));
    cudaMemcpyToSymbol(d_Vtl, &Vtl, sizeof(double));
    cudaMemcpyToSymbol(d_VHS, &VHS_coe, sizeof(double));
    cudaMemcpyToSymbol(d_Vmax, &Vmax, sizeof(double));
    cudaMemcpyToSymbol(d_Vstd, &Vstd, sizeof(double));
    cudaMemcpyToSymbol(d_tau, &tau, sizeof(double));
    cudaMemcpyToSymbol(d_Fn, &Fn, sizeof(size_t));
}

GPUCells::~GPUCells()
{
    cudaFree(d_particleNum);
    cudaFree(d_particleStartIndex);
    cudaFree(d_collisionCount);
    cudaFree(d_Temperature);
    cudaFree(d_Rho);
    cudaFree(d_Pressure);
    cudaFree(d_Velocity);
    cudaFree(d_ifCut);
    cudaFree(d_Segments);
}

void GPUCells::UploadFromHost(const int* h_particleNum, const int* h_particleStartIndex, const double& h_Unidx, const double& h_Unidy, const double& h_Unidz, const int* h_ifCut, const GPUSegment* h_Segments)
{
    cudaMemcpy(d_particleNum, h_particleNum, m_cellNum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleStartIndex, h_particleStartIndex, m_cellNum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ifCut, h_ifCut, m_cellNum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Segments, h_Segments, m_cellNum * sizeof(GPUSegment), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Unidx, &h_Unidx, sizeof(double));
    cudaMemcpyToSymbol(d_Unidy, &h_Unidy, sizeof(double));
    cudaMemcpyToSymbol(d_Unidz, &h_Unidz, sizeof(double));
    
}

void GPUCells::DownloadToHost(const double *h_Temperature, const double *h_Rho, const double *h_Pressure, const double3 *h_Velocity)
{
    size_t sized = m_cellNum * sizeof(double);
    size_t size3d = m_cellNum * sizeof(double3);
    cudaMemcpy((void*)h_Temperature, d_Temperature, sized, cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)h_Rho, d_Rho, sized, cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)h_Pressure, d_Pressure, sized, cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)h_Velocity, d_Velocity, size3d, cudaMemcpyDeviceToHost);
}

void GPUCells::CalparticleNum(const double3 *__restrict__ d_pos,
                              int *__restrict__ d_localID,
                              int *__restrict__ d_cellID, int particleNum,
                              int nx, int ny, int nz,
                              int blockSize)
{   
    cudaMemset(d_particleNum, 0, sizeof(int)*m_cellNum);
    int numBlocks = (particleNum + blockSize - 1) / blockSize;
    GPUCellKernels::CalparticleNumAndLocalID<<<numBlocks, blockSize>>>(d_pos, d_particleNum, d_localID, d_cellID, particleNum, nx, ny, nz);
}

void GPUCells::CalparticleStartIndex()
{
    thrust::device_ptr<int> d_particleNum_ptr = thrust::device_pointer_cast(d_particleNum);
    thrust::device_ptr<int> d_particleStartIndex_ptr = thrust::device_pointer_cast(d_particleStartIndex);

    thrust::exclusive_scan(
        d_particleNum_ptr, 
        d_particleNum_ptr + m_cellNum, 
        d_particleStartIndex_ptr
    );

    // for(int i = 0; i < m_cellNum; ++i){
    //     int particleNum = d_particleNum_ptr[i];
    //     int startIndex = d_particleStartIndex_ptr[i];
    //     printf("Cell %d: Particle Count = %d, Start Index = %d\n", i, particleNum, startIndex);
    // }
}

void GPUCells::Collision(double3* d_vel, int* global_id_sortted)
{
    int threadsPerBlock = 256;
    int numBlocks = (m_cellNum + threadsPerBlock - 1) / threadsPerBlock;
    
    GPUCellKernels::collisionInCells<<<numBlocks, threadsPerBlock>>>(
        d_vel,
        d_particleStartIndex,
        d_particleNum,
        m_cellNum,
        global_id_sortted,
        d_collisionCount
    );

    int totalCollisions = thrust::reduce(
        thrust::device_pointer_cast(d_collisionCount),
        thrust::device_pointer_cast(d_collisionCount + m_cellNum),
        0,
        thrust::plus<int>()
    );
    cudaDeviceSynchronize();  // 等待全部完成，便于调试和防止异步错误
    std::cout << "Collision Number is : " << totalCollisions << std::endl;
}

void GPUCells::Sample(const double3 *d_vel, int totalParticles, int* global_id_sortted)
{
    int blockSize = 256;
    int numBlocks = (m_cellNum + blockSize - 1) / blockSize;

    // 一个线程对应一个cell
    GPUCellKernels::samplingInCells<<<numBlocks, blockSize>>>(
        d_vel,
        d_particleStartIndex,
        d_particleNum,
        d_Temperature,
        d_Rho,
        d_Velocity,
        d_Pressure,
        m_cellNum,
        totalParticles,
        global_id_sortted
    );

    cudaDeviceSynchronize();
}

__global__ void GPUCellKernels::CalparticleNumAndLocalID(
    const double3* __restrict__ d_pos,
    int* __restrict__ d_particleNum,
    int* __restrict__ d_localID,
    int* __restrict__ d_cellID,
    int particleNum,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleNum) return;  // 保护越界

    double x = d_pos[idx].x;
    double y = d_pos[idx].y;
    double z = d_pos[idx].z;

    int i = static_cast<int>(x / d_Unidx);
    int j = static_cast<int>(y / d_Unidy);
    int k = static_cast<int>(z / d_Unidz);

    if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz) {
        int cellID = k + j * nz + i * ny * nz;

        // atomicAdd的返回值就是本粒子的local id
        int local = atomicAdd(&(d_particleNum[cellID]), 1);

        d_localID[idx] = local;  // 记录下来
        d_cellID[idx] = cellID;
        // printf("Particle %d assigned to cell %d with local ID %d\n", idx, cellID, local);
    }
}

__global__ void GPUCellKernels::collisionInCells(
    double3* d_vel,
    const int* __restrict__ d_particleStartIndex,
    const int* __restrict__ d_particleNum,
    int totalCells,
    int* global_id_sortted,
    int* d_collisionCount
) {
    int cellID = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellID >= totalCells) return;

    int start = d_particleStartIndex[cellID];
    int np = d_particleNum[cellID];

    if (np < 2) return;

    // 初始化随机数状态器
    curandState localState;
    curand_init(clock64() + cellID * 1234, 0, 0, &localState);

    // Cell体积
    double cell_volume = d_Unidx * d_Unidy * d_Unidz;

    double Srcmax = d_Vmax * M_PI * diam * diam;
    // 计算预计碰撞次数
    double expected_Mcand = static_cast<double>(np * np) * Fn * Srcmax * tau / (2.0 * cell_volume);
    int M_candidate = static_cast<int>(expected_Mcand);

    int local_collision_count {};
    for (int m = 0; m < M_candidate; ++m) {
        // 随机挑选两个不同粒子
        int idx1 = start + (curand(&localState) % np);
        int idx2 = start + (curand(&localState) % np);
        while (idx1 == idx2) {
            idx2 = start + (curand(&localState) % np);
        }
        int idx1_sorted = global_id_sortted[idx1];
        int idx2_sorted = global_id_sortted[idx2];

        // 读取粒子速度
        double3 v1 = d_vel[idx1_sorted];
        double3 v2 = d_vel[idx2_sorted];

        // 相对速度
        double3 v_rel = make_double3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
        double v_rel_mag = sqrt(v_rel.x * v_rel.x + v_rel.y * v_rel.y + v_rel.z * v_rel.z);

        // 碰撞截面
        double Src = v_rel_mag * d_diam * d_diam * M_PI * pow((2.0 * d_boltz * T / (0.5 * mass * v_rel_mag * v_rel_mag)), d_Vtl-0.5) / d_VHS;
        Srcmax = fmax(Src, Srcmax);

        double rand01 = curand_uniform(&localState);
        if (rand01 < Src / Srcmax) {
            // 碰撞发生，重新采样相对速度方向
            double cosr = 2.0 * curand_uniform(&localState) - 1.0;
            double sinr = sqrt(1.0 - cosr * cosr);
            double phi = 2.0 * M_PI * curand_uniform(&localState);

            double3 vrel_new = make_double3(
                cosr,
                sinr * cos(phi),
                sinr * sin(phi)
            );

            double scale = v_rel_mag;
            vrel_new.x *= scale;
            vrel_new.y *= scale;
            vrel_new.z *= scale;

            double3 Vmean = make_double3(
                0.5 * (v1.x + v2.x),
                0.5 * (v1.y + v2.y),
                0.5 * (v1.z + v2.z)
            );

            // 更新粒子速度
            d_vel[idx1_sorted].x = Vmean.x + 0.5 * vrel_new.x;
            d_vel[idx1_sorted].y = Vmean.y + 0.5 * vrel_new.y;
            d_vel[idx1_sorted].z = Vmean.z + 0.5 * vrel_new.z;

            d_vel[idx2_sorted].x = Vmean.x - 0.5 * vrel_new.x;
            d_vel[idx2_sorted].y = Vmean.y - 0.5 * vrel_new.y;
            d_vel[idx2_sorted].z = Vmean.z - 0.5 * vrel_new.z;

            local_collision_count ++;
        }
    }

    d_collisionCount[cellID] = local_collision_count;
}


__global__ void GPUCellKernels::samplingInCells(
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
) {
    int cellID = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellID >= totalCells) return;

    int start = d_particleStartIndex[cellID];
    int np = d_particleNum[cellID];
    double cell_volume = d_Unidx * d_Unidy * d_Unidz; // 均匀网格

    if (np == 0) {
        d_Temperature[cellID] = 0.0;
        d_Rho[cellID] = 0.0;
        d_Velocity[cellID] = make_double3(0.0, 0.0, 0.0);
        d_Pressure[cellID] = 0.0;
        return;
    }
    
    double vx_sum = 0.0;
    double vy_sum = 0.0;
    double vz_sum = 0.0;
    double E_sum = 0.0;

    for (int i = 0; i < np; ++i) {
        int idx = start + i;
        if (idx >= totalParticles) continue;

        int idx_sortted {global_id_sortted[idx]};
        double vx = d_vel[idx_sortted].x;
        double vy = d_vel[idx_sortted].y;
        double vz = d_vel[idx_sortted].z;

        vx_sum += vx;
        vy_sum += vy;
        vz_sum += vz;
        E_sum += 0.5 * (vx * vx + vy * vy + vz * vz);
    }

    double inv_np = 1.0 / static_cast<double>(np);
    double3 v_avg = make_double3(
        vx_sum * inv_np,
        vy_sum * inv_np,
        vz_sum * inv_np
    );

    double v_avg_sq = 0.5 * (v_avg.x * v_avg.x + v_avg.y * v_avg.y + v_avg.z * v_avg.z);
    double E_avg = E_sum * inv_np;

    double Rho = static_cast<double>(np * d_Fn) * mass / cell_volume;
    double T = mass * (2.0/3.0) * (E_avg - v_avg_sq) / d_boltz;
    double Pressure = (2.0/3.0) * (E_avg - v_avg_sq) * Rho;

    d_Velocity[cellID] = v_avg;
    d_Temperature[cellID] = T;
    d_Rho[cellID] = Rho;
    d_Pressure[cellID] = Pressure;
}
    
    