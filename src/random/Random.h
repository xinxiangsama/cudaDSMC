#pragma once
#include <random>
#include <Eigen/Dense>

class Random
{
public:
    Random();
    ~Random() = default;

    void modifygenerator(std::mt19937 &generator);
    void modifygenerator(const int& rd);
    double getrandom01();
    int getrandomint(const int& min, const int& max);

    Eigen::Vector3d MaxwellDistribution(const double& Vstd);
    double RotationalenergySample();

protected:
    std::random_device m_rd;   // Random device for seeding
    std::mt19937 m_mt19937;    // Mersenne Twister RNG
    std::uniform_real_distribution<double> m_dist01; // Uniform distribution
};
