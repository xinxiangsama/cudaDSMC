// GPUBoundary.cuh
#pragma once
#include <cuda_runtime.h>
#include "Param.h"
#include <math.h> 

namespace GPUBoundary {
    enum BoundaryType {
        PERIODIC = 0,
        WALL = 1,
    };
    struct Boundary {
        double3 point;   // 边界上的一个点
        double3 normal;  // 边界的法向
        BoundaryType type;
    };
    /*pos: particle position, point: a point on the boundary, normal: the normal of boundary*/
    __device__ __forceinline__  bool isHit(const double3& pos, const double3& point, const double3& normal) {
        double dx = pos.x - point.x;
        double dy = pos.y - point.y;
        double dz = pos.z - point.z;
        double dot = dx * normal.x + dy * normal.y + dz * normal.z;
        return dot < 0.0;
    };

    namespace PeriodicBoundary {

        __device__ __forceinline__  void apply(
            double3& pos, const double3& point, const double3& normal
        ){
            pos.x += normal.x * L1;
            pos.y += normal.y * L2;
            pos.z += normal.z * L3;

        }
        

    }  // namespace PeriodicBoundary
    namespace WallBoundary {

    }// namespace WallBoundary
}  // namespace GPUBoundary
