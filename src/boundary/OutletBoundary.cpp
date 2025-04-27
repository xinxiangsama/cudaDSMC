#include "OutletBoundary.h"

OutletBoundary::OutletBoundary(const Particle::Coord &point, const Particle::Coord &normal) : m_point(point), m_normal(normal.normalized())
{
}

bool OutletBoundary::isHit(const Particle::Coord &position) const
{
    double distance = (position - m_point).dot(m_normal);
    return distance < 0.0;
}

void OutletBoundary::Reflect(Particle *particle, const double &dt) const
{
    particle->setInvalid();
}
