#pragma once
#include "Boundary.h"

class WallBoundary : public Boundary
{
public:
    WallBoundary(const Particle::Coord& point, const Particle::Coord& normal, bool diffuse = false);
    virtual ~WallBoundary() = default;

    virtual bool isHit(const Particle::Coord& position) const override;
    virtual void Reflect(Particle* particle, const double& dt) const override;
    virtual const Particle::Coord& getnormal() const override { return m_normal; }

private:
    Particle::Coord m_point;   // a point in boundary
    Particle::Coord m_normal;  // normal
    bool m_diffuse;            // if diffuse
};
 
