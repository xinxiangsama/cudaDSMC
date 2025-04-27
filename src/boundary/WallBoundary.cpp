#include "WallBoundary.h"
#include <memory>
extern double Vstd;
extern std::unique_ptr<Random> randomgenerator;

WallBoundary::WallBoundary(const Particle::Coord &point, const Particle::Coord &normal, bool diffuse) : m_point(point), m_normal(normal.normalized()), m_diffuse(diffuse)
{
}

bool WallBoundary::isHit(const Particle::Coord &position) const
{
    // Check if the particle is within a certain distance from the wall
    double distance = (position - m_point).dot(m_normal);
    return distance < 0.0;
}

void WallBoundary::Reflect(Particle* particle, const double& dt) const
{   
    auto velocity = particle->getvelocity();
    auto position = particle->getposition() - velocity * dt; //back to previous position
    auto distance = (position - m_point).dot(m_normal);
    auto v_normal = velocity.dot(m_normal);
    if(v_normal >= 0.0){
        return;
    }
    auto t_hit = - distance / v_normal;
    if(t_hit > dt || t_hit < 0.0){
        return;
    }

    auto hit_position = position + velocity * t_hit;

    if(m_diffuse){
        // Diffuse reflection
        auto rand01 = randomgenerator->getrandom01();
        auto V = sqrt(-log(rand01)) * Vstd;
        rand01 = randomgenerator->getrandom01();
        auto u = V * cos(2 * M_PI * rand01);
        auto w = V * sin(2 * M_PI * rand01);
        rand01 = randomgenerator->getrandom01();
        auto v = sqrt(-log(rand01)) * Vstd;
        auto new_velocity = Eigen::Vector3d(u, v * m_normal(1), w);
        particle->setvelocity(new_velocity);
    }else{
        // Specular reflection
        auto new_velocity = velocity  + 2 * (abs(velocity.dot(m_normal))) * m_normal;
        particle->setvelocity(new_velocity);
    }
    // Update the position of the particle
    auto new_position = hit_position + particle->getvelocity() * (dt - t_hit);
    particle->setposition(new_position);
}
