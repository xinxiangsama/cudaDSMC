#include "Segment.h"
#include "random/Random.h"
#include <iostream>
#include <memory>
#include "Param.h"
extern std::unique_ptr<Random> randomgenerator;

bool Segment::isHit(const Particle::Coord &position) const
{   
    Particle::Coord m_point = 0.5 * (m_leftpoint->getPosition() + m_rightpoint->getPosition());
    // Particle::Coord m_point = m_rightpoint->getPosition();
    // Check if the particle is within a certain distance from the wall
    double distance = (position - m_point).dot(m_normal);
    return distance < 0.0;
    // return true;
}

void Segment::Reflect(Particle* particle, const double &dt) const
{
    Particle::Coord m_point = 0.5 * (m_leftpoint->getPosition() + m_rightpoint->getPosition());
    auto velocity = particle->getvelocity();
    auto position = particle->getposition();
    position -= velocity * dt; //back to previous position
    auto distance = (position - m_point).dot(m_normal);
    auto v_normal = velocity.dot(m_normal);
    if(v_normal > 0.0){
        return;
    }
    auto t_hit = abs( - distance / v_normal);
    t_hit = std::min(t_hit, dt);
    if(t_hit > dt || t_hit < 0.0){
        return;
    }

    auto hit_position = position + velocity * t_hit;
    // if(pow(hit_position.x() - Center_x, 2) + pow(hit_position.y() - Center_y, 2) < (Radius * Radius)){
    //     std::cerr <<"hit position is in cylinder"<<std::endl;
    // }
    // // Specular reflection
    auto new_velocity = velocity  + 2 * abs(v_normal) * m_normal;
    particle->setvelocity(new_velocity);
    // // Update the position of the particle
    // particle->setvelocity(V_jet * m_normal);
    auto new_position = hit_position + particle->getvelocity() * abs(dt - t_hit);
    // particle->setposition(position);
    particle->setposition(new_position);
    // particle->setposition({999, 999, 999});

    // particle->setvelocity(abs(velocity.dot(m_normal)) * m_normal);
    // particle->setvelocity(V_jet * m_normal);
    // particle->setvelocity({0,0,0});
    // particle->setposition(position + m_normal * 1e-5);
}

const double &Segment::getlength() const
{
    return m_length;
}

const LargrangianPoint::Coord &Segment::getsloop() const
{
    return m_sloop;
}

const LargrangianPoint::Coord &Segment::getnormal() const
{
    return m_normal;
}

LargrangianPoint *Segment::getleftpoint() const
{
    return m_leftpoint;
}

LargrangianPoint *Segment::getrightpoint() const
{
    return m_rightpoint;
}

void Segment::setlength(const double &length)
{
    m_length = length;
}

void Segment::setsloop(const Coord &sloop)
{
    m_sloop = sloop;
}

void Segment::setnormal(const Coord &normal)
{
    m_normal = normal;
}

void Segment::setleftpoint(LargrangianPoint *leftpoint)
{
    m_leftpoint = leftpoint;
}

void Segment::setrightpoint(LargrangianPoint *rightpoint)
{
    m_rightpoint = rightpoint;
}