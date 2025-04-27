#include "PeriodicBoundary.cuh"
#include "Param.h"

__device__ void GPUBoundary::PeriodicBoundary::apply(
    double3& pos, const double3& point, const double3& normal
) {
    double3 diff = make_double3(
        pos.x - point.x,
        pos.y - point.y,
        pos.z - point.z
    );

    double dot = diff.x * normal.x + diff.y * normal.y + diff.z * normal.z;

    if (dot > 0.0) {
        pos.x -= L1 * normal.x;
        pos.y -= L2 * normal.y;
        pos.z -= L3 * normal.z;
    } else if (dot < 0.0) {
        pos.x += L1 * normal.x;
        pos.y += L2 * normal.y;
        pos.z += L3 * normal.z;
    }
}
