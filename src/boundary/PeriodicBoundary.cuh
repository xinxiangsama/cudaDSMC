#include "Boundary.cuh"

namespace GPUBoundary {
    namespace PeriodicBoundary {

        __device__ void apply(
            double3& pos, const double3& point, const double3& normal
        );

    }  // namespace PeriodicBoundary
}  // namespace GPUBoundary
