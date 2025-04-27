#include "Boundary.cuh"

    /*pos: particle position, point: a point on the boundary, normal: the normal of boundary*/
    __device__ inline bool GPUBoundary::isHit(const double3& pos, const double3& point, const double3& normal) {
        double dx = pos.x - point.x;
        double dy = pos.y - point.y;
        double dz = pos.z - point.z;
        double dot = dx * normal.x + dy * normal.y + dz * normal.z;
        return dot < 0.0;
    }
