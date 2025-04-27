#pragma once
#include "Boundary.h"
#include "../random/Random.h"
#include <memory>

class InletBoundary : public Boundary
{
public:
    InletBoundary() = default;
    InletBoundary(const int& _numprocs) : numprocs(_numprocs) {};
    virtual ~InletBoundary() = default;

    virtual void InjetParticle(std::vector<Particle>& particles) override;

    virtual bool isHit(const Particle::Coord& position) const;

    virtual void Reflect(Particle* particle, const double& dt) const;
protected:
    int numprocs;
};