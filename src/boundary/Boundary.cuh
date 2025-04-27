// GPUBoundary.cuh
#pragma once
#include <cuda_runtime.h>
#include <math.h>  // for fmod

namespace GPUBoundary {
    /*pos: particle position, point: a point on the boundary, normal: the normal of boundary*/
    __device__ inline bool isHit(const double3& pos, const double3& point, const double3& normal);
}  // namespace GPUBoundary
