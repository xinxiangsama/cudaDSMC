#include "InletBoundary.h"
extern double Vstd;
extern std::unique_ptr<Random> randomgenerator;

void InletBoundary::InjetParticle(std::vector<Particle>& particles)
{
    double JetLength = V_jet * tau;
    double JetVolume = JetLength * L2 * L3;
    size_t JetParticleNum = ((JetVolume * Rho / mass) / Fn);
    JetParticleNum /= numprocs; // assign to every procs;

    for(int i = 0; i < JetParticleNum; ++i){
        auto rx = randomgenerator->getrandom01();
        auto ry = randomgenerator->getrandom01();
        auto rz = randomgenerator->getrandom01();
        double x = JetLength * rx;
        double y = L2 * ry;
        double z = L3 * rz;
        auto velocity = randomgenerator->MaxwellDistribution(Vstd);
        velocity(0) += V_jet;
        particles.emplace_back(mass, Eigen::Vector3d{x, y, z}, velocity);
    }

}

bool InletBoundary::isHit(const Particle::Coord &position) const
{
    return false;
}

void InletBoundary::Reflect(Particle *particle, const double &dt) const
{
}
