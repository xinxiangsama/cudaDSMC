#include "Particle.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <curand_kernel.h>



extern __constant__ double d_Unidx;
extern __constant__ double d_Unidy;
extern __constant__ double d_Unidz;
extern __constant__ double d_Vstd;

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
	cudaMalloc((void**)&global_id_sortted, sizeints);
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
	cudaFree(global_id_sortted);
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

void GPUParticles::Move(const double &dt, const double &blockSize, const Boundary* d_boundaries)
{
    int numBlocks = (N + blockSize - 1) / blockSize;
    GPUParticleKernels::moveParticles<<<numBlocks, blockSize>>>(d_pos_x, d_pos_y, d_pos_z,
                                          d_vel_x, d_vel_y, d_vel_z,
                                          N, dt, d_boundaries);
    cudaDeviceSynchronize();
}

void GPUParticles::Sort(const int* d_particleStartIndex)
{	
	int blockSize = 128;
	int numBlocks = (N + blockSize - 1) / blockSize;
	GPUParticleKernels::sortParticles<<<numBlocks, blockSize>>>(cell_id, local_id, global_id, global_id_sortted, d_particleStartIndex, N);
}

__global__ void GPUParticleKernels::moveParticles(double* pos_x, double* pos_y, double* pos_z,
                               double* vel_x, double* vel_y, double* vel_z,
                               int N, double dt, const Boundary* d_boundaries) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) 
        return;

    pos_x[i] += vel_x[i] * dt;
    pos_y[i] += vel_y[i] * dt;  

    /*============边界条件=============*/
    /*x方向*/
    if(pos_x[i] < 0){
		// pos_x[i] = fmod(pos_x[i], L1) + L1;
        auto pos = make_double3(pos_x[i], pos_y[i], pos_z[i]);
        // auto point = make_double3(0, 0.5 * L2, 0.5 * L3);
        // GPUBoundary::PeriodicBoundary::apply(pos, point, make_double3(1, 0, 0));
        GPUBoundary::PeriodicBoundary::apply(pos, d_boundaries[0].point, d_boundaries[0].normal);
        pos_x[i] = pos.x;
	}

	if(pos_x[i] > L1){
		// pos_x[i] = fmod(pos_x[i], L1);
        auto pos = make_double3(pos_x[i], pos_y[i], pos_z[i]);
        // auto point = make_double3(L1, 0.5 * L2, 0.5 * L3);
        // GPUBoundary::PeriodicBoundary::apply(pos, point, make_double3(-1, 0, 0));
        GPUBoundary::PeriodicBoundary::apply(pos, d_boundaries[1].point, d_boundaries[1].normal);
        pos_x[i] = pos.x;
	}
	/*y方向*/
	if (pos_y[i] < 0) {
		// pos_y[i] = fmod(pos_y[i], L2) + L2;
        auto pos = make_double3(pos_x[i], pos_y[i], pos_z[i]);
        auto vel = make_double3(vel_x[i], vel_y[i], vel_z[i]);
        auto point = make_double3(0.5 * L1, 0.0, 0.5 * L3);
        GPUBoundary::WallBoundary::apply(pos, vel, point, make_double3(0, 1, 0));
        pos_y[i] = pos.y;
        vel_y[i] = vel.y;
	}

	if (pos_y[i] > L2) {
		// pos_y[i] = fmod(pos_y[i], L2);
        auto pos = make_double3(pos_x[i], pos_y[i], pos_z[i]);
        auto vel = make_double3(vel_x[i], vel_y[i], vel_z[i]);
        auto point = make_double3(0.5 * L1, L2, 0.5 * L3);
        GPUBoundary::WallBoundary::apply(pos, vel, point, make_double3(0, -1, 0));
        pos_y[i] = pos.y;
        vel_y[i] = vel.y;
        vel_x[i] = 300;
	}

    // pos_z[i] += vel_z[i] * dt;                       
    // // 检查所有边界
    // for (int b = 0; b < 6; ++b) {
    //     const Boundary& boundary = d_boundaries[b];
    //     if (GPUBoundary::isHit(pos, boundary.point, boundary.normal)) {
    //         // if (boundary.type == GPUBoundary::BoundaryType::PERIODIC)
    //             // GPUBoundary::PeriodicBoundary::apply(pos, boundary.point, boundary.normal);
    //     }
    // }

}

__global__ void GPUParticleKernels::sortParticles(const int *cell_id, const int *local_id, const int *global_id, int *global_id_sortted, const int *d_particleStartIndex, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) 
        return;
	
	int sorted_global_id = d_particleStartIndex[cell_id[i]] + local_id[i];
	global_id_sortted[sorted_global_id] = global_id[i];
}
