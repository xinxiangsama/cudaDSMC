#pragma once
#include <cuda_runtime.h>
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

    __device__ void Reflect(double3& pos, double3& vel, const double& dt) {
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

        // Specular reflection: update velocity
        double3 new_velocity = make_double3(
            vel.x + 2.0 * fabs(v_normal) * m_normal.x,
            vel.y + 2.0 * fabs(v_normal) * m_normal.y,
            vel.z + 2.0 * fabs(v_normal) * m_normal.z
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
    double3 m_point{};
    double3 m_normal{};
};
