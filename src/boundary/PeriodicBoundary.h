#pragma once
#include "Boundary.h"

class PeriodicBoundary : public Boundary
{
public:
    PeriodicBoundary() = default;
    PeriodicBoundary(const Particle::Coord& point, const Particle::Coord& normal,const int& direction, const double & Length)
        : m_point(point), m_normal(normal.normalized()), m_direction(direction), m_length(Length) {};

    bool isHit(const Particle::Coord& position) const override;

    void Reflect(Particle* particle, const double& dt) const override;
protected:
    Particle::Coord m_point;   // a point in boundary
    int m_direction;
    Particle::Coord m_normal;  // normal
    double m_length;          // length of the domain
};