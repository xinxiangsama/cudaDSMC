#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "Param.h"
#include "Segment.h"
class GPUSegment
{
public:
    GPUSegment() = default;
    GPUSegment(const double3& point, const double3& normal): m_point(point), m_normal(normal){};
    GPUSegment(Segment* segment){
        auto leftpoint {segment->getleftpoint()->getPosition()};
        auto rightpoint {segment->getrightpoint()->getPosition()};
        auto normal {segment->getnormal()};
        m_point = make_double3(0.5 * (leftpoint.x() + rightpoint.x()),
                       0.5 * (leftpoint.y() + rightpoint.y()),
                       0.5 * (leftpoint.z() + rightpoint.z()));
        m_normal = make_double3(normal.x(), normal.y(), normal.z());
    }
    __device__ bool isHit(const double3& pos) {
        auto vector = make_double3(pos.x - m_point.x, pos.y - m_point.y, pos.z - m_point.z);
        double distance = vector.x * m_normal.x +
                          vector.y * m_normal.y +
                          vector.z * m_normal.z;
        return distance < 0;
    }

    __device__ void Reflect(double3& pos, double3& vel, const double& dt, curandState& state) {
        // Calculate the distance to the plane
        double3 vector = make_double3(pos.x - m_point.x, pos.y - m_point.y, pos.z - m_point.z);
        double distance = vector.x * m_normal.x +
                          vector.y * m_normal.y +
                          vector.z * m_normal.z;

        // Calculate the normal velocity component
        double v_normal = vel.x * m_normal.x +
                          vel.y * m_normal.y +
                          vel.z * m_normal.z;

        if (v_normal > 0.0) {
            return; // No reflection if moving away from the plane
        }

        // Calculate the time of hit
        double t_hit = fabs(-distance / v_normal);
        t_hit = fmin(t_hit, dt);

        if (t_hit > dt || t_hit < 0.0) {
            return; // No valid hit within the time step
        }

        // Calculate the hit position
        double3 hit_position = make_double3(
            pos.x + vel.x * t_hit,
            pos.y + vel.y * t_hit,
            pos.z + vel.z * t_hit
        );

        // Diffuse reflection: sample velocity based on Maxwell velocity distribution
        double3 tang1, tang2;
        if (fabs(m_normal.x) > 0.1)
            tang1 = normalizeVector(cross(make_double3(0.0, 1.0, 0.0), m_normal));
        else
            tang1 = normalizeVector(cross(make_double3(1.0, 0.0, 0.0), m_normal));
        tang2 = normalizeVector(cross(m_normal, tang1));

        // Thermal velocity sampling using GPU random number generator

        double rand1 = curand_uniform_double(&state);
        double V = sqrt(-log(rand1)) * sqrt(2.0 * boltz * T_wall / mass);

        double theta = 2.0 * M_PI * curand_uniform_double(&state);
        double v = V * cos(theta);
        double w = V * sin(theta);
        double u = V;  // Normal direction (positive)

        // Assemble new velocity in local coordinates and map back to global coordinates
        double3 new_velocity = make_double3(
            u * m_normal.x + v * tang1.x + w * tang2.x,
            u * m_normal.y + v * tang1.y + w * tang2.y,
            u * m_normal.z + v * tang1.z + w * tang2.z
        );

        vel = new_velocity;

        // Update the position of the particle
        double remaining_time = dt - t_hit;
        double3 new_position = make_double3(
            hit_position.x + vel.x * remaining_time,
            hit_position.y + vel.y * remaining_time,
            hit_position.z + vel.z * remaining_time
        );

        pos = new_position;
    }
protected:
    __device__ double3 normalizeVector(const double3& vec) {
        double magnitude = sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        return make_double3(vec.x / magnitude, vec.y / magnitude, vec.z / magnitude);
    }

    __device__ double3 cross(const double3& a, const double3& b) {
        return make_double3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    double3 m_point{};
    double3 m_normal{};
};
