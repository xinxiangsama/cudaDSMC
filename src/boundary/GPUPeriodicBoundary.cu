#include "PeriodicBoundary.cuh"

__device__ void GPUBoundary::PeriodicBoundary::reflect(double3 &pos, int direction, double length)
{
    double* coord = (direction == 0) ? &pos.x :
    (direction == 1) ? &pos.y : &pos.z;

    *coord = (*coord < 0.0) ? fmod(*coord, length) + length
                : fmod(*coord, length);
}