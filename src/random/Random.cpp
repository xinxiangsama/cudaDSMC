#include "Random.h"

Random::Random()
    : m_mt19937(m_rd()),       // Seed mt19937 with random_device
      m_dist01(0.0, 1.0)         // Initialize uniform distribution
{}

void Random::modifygenerator(std::mt19937 &generator)
{
    m_mt19937 = generator;  // Modify the generator
}

void Random::modifygenerator(const int &rd)
{
    m_mt19937.seed(rd); 
}

double Random::getrandom01()
{
    return m_dist01(m_mt19937);  // Use the distribution
}

int Random::getrandomint(const int &min, const int &max)
{
    std::uniform_int_distribution<int> dist(min, max);
    return dist(m_mt19937);  // Use the distribution
}

Eigen::Vector3d Random::MaxwellDistribution(const double &Vstd)
{
    auto rd1 = getrandom01();
    auto rd2 = getrandom01();

    auto u = sqrt(-log(rd1)) * sin(2 * M_PI * rd2) * Vstd;

    rd1 = getrandom01();
    rd2 = getrandom01();
    auto v = sqrt(-log(rd1)) * sin(2 * M_PI * rd2) * Vstd;

    rd1 = getrandom01();
    rd2 = getrandom01();
    auto w = sqrt(-log(rd1)) * sin(2 * M_PI * rd2) * Vstd;
    
    return Eigen::Vector3d(u, v, w);
}
