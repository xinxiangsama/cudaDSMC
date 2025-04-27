#include "Boundary.cuh"

namespace GPUBoundary {
namespace PeriodicBoundary {

__device__ inline void reflect(double3& pos, int direction, double length);

}  // namespace PeriodicBoundary
}  // namespace GPUBoundary
