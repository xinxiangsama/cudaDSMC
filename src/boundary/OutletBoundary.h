#pragma once
#include "Boundary.h"
#include "../random/Random.h"
#include <memory>

class OutletBoundary : public Boundary
{
public:
    OutletBoundary() = default;
    virtual ~OutletBoundary() = default;
    OutletBoundary (const Particle::Coord& point, const Particle::Coord& normal);

    virtual bool isHit(const Particle::Coord& position) const;

    virtual void Reflect(Particle* particle, const double& dt) const;
    virtual const Particle::Coord& getnormal() const override { return m_normal; }
protected:
    Particle::Coord m_point;   // a point in boundary
    Particle::Coord m_normal;  // normal
};