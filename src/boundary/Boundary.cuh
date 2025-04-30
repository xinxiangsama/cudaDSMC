// GPUBoundary.cuh
#pragma once
#include <cuda_runtime.h>
#include "Param.h"
#include <curand_kernel.h>
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
        __device__ __forceinline__ void apply(
            double3& pos, double3& vel,
            const double3& point, const double3& normal
        ) {
            // Step 1: 计算从边界点到粒子位置的向量
            double dx = pos.x - point.x;
            double dy = pos.y - point.y;
            double dz = pos.z - point.z;
    
            // Step 2: 计算法向分量（点乘）
            double dot = dx * normal.x + dy * normal.y + dz * normal.z;
    
            // Step 3: 位置反射 —— 镜像到边界另一侧
            pos.x -= 2.0 * dot * normal.x;
            pos.y -= 2.0 * dot * normal.y;
            pos.z -= 2.0 * dot * normal.z;
    
            // Step 4: 速度反射 —— 法向速度翻转，切向速度不变
            double vdot = vel.x * normal.x + vel.y * normal.y + vel.z * normal.z;
            vel.x -= 2.0 * vdot * normal.x;
            vel.y -= 2.0 * vdot * normal.y;
            vel.z -= 2.0 * vdot * normal.z;
        }
    
    }// namespace WallBoundary
    namespace OutflowBoundary{
        __device__ __forceinline__ void apply()
        {

        }
    }// namespace OutflowBoundary
}  // namespace GPUBoundary