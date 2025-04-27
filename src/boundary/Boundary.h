#pragma once
#include "../particle/Particle.h"
#include "../Param.h"
#include <memory>

class Boundary
{
public:
    Boundary() = default;
    virtual ~Boundary() = default;

    // judge the particle whether arrive the boundary
    virtual bool isHit(const Particle::Coord& position) const = 0;

    // Injet particle
    virtual void InjetParticle(std::vector<Particle>& particles) {};

    // define the action of particle when it hit the boundary
    virtual void Reflect(Particle* particle, const double& dt) const = 0;

    virtual bool isPeriodic() const { return false; }

    virtual const Particle::Coord& getnormal() const {}
};
