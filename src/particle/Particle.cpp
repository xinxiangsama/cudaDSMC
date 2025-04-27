#include "Particle.h"
#include <memory>

extern std::unique_ptr<Random> randomgenerator;

const double &Particle::getmass()
{
    return m_mass;
}

const Particle::Coord &Particle::getposition()
{
    return m_position;
}
const Particle::Coord &Particle::getvelocity()
{
    return m_velocity;
}
const int &Particle::getcellID()
{
    return m_cellID;
}
const int &Particle::getlocalID()
{
    return m_localID;
}
const int &Particle::getglobalID()
{
    return m_globalID;
}
const double &Particle::gettmove()
{
    return t_move;
}
void Particle::setmass(const double &mass)
{
    m_mass = mass;
}
void Particle::setposition(const Coord &position)
{
    m_position = position;
}
void Particle::setvelocity(const Coord &velocity)
{
    m_velocity = velocity;
}

void Particle::settmove(const double &t)
{
    t_move = t;
}

void Particle::setcellID(const int &id)
{
    m_cellID = id;
}

void Particle::setlocalID(const int &id)
{
    m_localID = id;
}

void Particle::setglobalID(const int &id)
{
    m_globalID = id;
}

void Particle::Move(const double &dt)
{
    m_position += m_velocity * dt;
}

void Particle::Collision(Particle *other)
{
    Particle::Coord v_mean = static_cast<Particle::Coord>(0.5 * (m_velocity + other->getvelocity()));
    Particle::Coord v_rel = static_cast<Particle::Coord>(m_velocity - other->getvelocity());
    auto v_rel_mag = v_rel.norm();
    auto cosr = 2.0 * randomgenerator->getrandom01() - 1.0;
    auto sinr = sqrt(1.0 - cosr * cosr);
    auto phi = 2.0 * M_PI * randomgenerator->getrandom01();

    Particle::Coord vrel_new(
        cosr,
        sinr * cos(phi),
        sinr * sin(phi)
    );

    vrel_new = vrel_new.normalized() * v_rel_mag;

    m_velocity = v_mean + 0.5 * vrel_new;
    other->setvelocity(v_mean - 0.5 * vrel_new);
}
