#include "PeriodicBoundary.h"

bool PeriodicBoundary::isHit(const Particle::Coord &position) const
{
    // Check if the particle is within a certain distance from the periodic boundary
    double distance = (position - m_point).dot(m_normal);
    return distance < 0.0;
}

void PeriodicBoundary::Reflect(Particle *particle, const double &dt) const
{
    auto position = particle->getposition();

    double& coord = position[m_direction];

    if(coord < 0.0){
        coord = std::fmod(coord, m_length) + m_length;
    }else{
        coord = std::fmod(coord, m_length);
    }
    particle->setposition(position);
}
