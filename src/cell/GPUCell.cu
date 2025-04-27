#include "Cell.cuh"
#include "Param.h"
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
__constant__ double d_Unidx;
__constant__ double d_Unidy;
__constant__ double d_Unidz;
__constant__ double d_L1;
__constant__ double d_L2;
__constant__ double d_L3;

GPUCells::GPUCells(const int& N) : m_cellNum(N)
{
    size_t sizei = N * sizeof(int);
    size_t sized = N * sizeof(double);
    size_t size3d = N * sizeof(double3);

    cudaMalloc((void**)&d_particleNum, sizei);
    cudaMalloc((void**)&d_particleStartIndex, sizei);
    cudaMalloc((void**)&d_Temperature, sized);
    cudaMalloc((void**)&d_Rho, sized);
    cudaMalloc((void**)&d_Pressure, sized);
    cudaMalloc((void**)&d_Velocity, size3d);
}

GPUCells::~GPUCells()
{
    cudaFree(d_particleNum);
    cudaFree(d_particleStartIndex);
    cudaFree(d_Temperature);
    cudaFree(d_Rho);
    cudaFree(d_Pressure);
    cudaFree(d_Velocity);
}

void GPUCells::UploadFromHost(const int* h_particleNum, const int* h_particleStartIndex, const double& h_Unidx, const double& h_Unidy, const double& h_Unidz)
{
    cudaMemcpy(d_particleNum, h_particleNum, m_cellNum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleStartIndex, h_particleStartIndex, m_cellNum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Unidx, &h_Unidx, sizeof(double));
    cudaMemcpyToSymbol(d_Unidy, &h_Unidy, sizeof(double));
    cudaMemcpyToSymbol(d_Unidz, &h_Unidz, sizeof(double));
    cudaMemcpyToSymbol(d_L1, &L1, sizeof(double));
    cudaMemcpyToSymbol(d_L2, &L2, sizeof(double));
    cudaMemcpyToSymbol(d_L2, &L2, sizeof(double));
    
}

void GPUCells::CalparticleNum(const double *__restrict__ d_pos_x, 
    const double *__restrict__ d_pos_y, 
    const double *__restrict__ d_pos_z, 
    int *__restrict__ d_localID, 
    int *__restrict__ d_cellID, int particleNum, 
    int nx, int ny, int nz, 
    int blockSize)
{   
    cudaMemset(d_particleNum, 0, sizeof(int)*m_cellNum);
    int numBlocks = (particleNum + blockSize - 1) / blockSize;
    GPUCellKernels::CalparticleNumAndLocalID<<<numBlocks, blockSize>>>(d_pos_x, d_pos_y, d_pos_z, d_particleNum, d_localID, d_cellID, particleNum, nx, ny, nz);
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


__global__ void GPUCellKernels::CalparticleNumAndLocalID(
    const double* __restrict__ d_pos_x,
    const double* __restrict__ d_pos_y,
    const double* __restrict__ d_pos_z,
    int* __restrict__ d_particleNum,
    int* __restrict__ d_localID,
    int* __restrict__ d_cellID,
    int particleNum,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleNum) return;  // 保护越界

    double x = d_pos_x[idx];
    double y = d_pos_y[idx];
    double z = d_pos_z[idx];

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