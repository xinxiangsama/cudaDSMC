#pragma once
#include <Eigen/Dense>

class Phase
{
public:
    using Coord = Eigen::Vector3d;
    Phase() = default;
    virtual ~Phase() = default;

    // Accessors
    const double& getdensity() const { return m_density; }
    const Coord& getvelocity() const { return m_velocity; }
    const double& getpressure() const { return m_pressure; }
    const double& gettemperature() const { return m_temperature; }

    // Modifiers
    void setdensity(const double& density) { m_density = density; }
    void setvelocity(const Coord& velocity) { m_velocity = velocity; }
    void setpressure(const double& pressure) { m_pressure = pressure; }
    void settemperature(const double& temperature) { m_temperature = temperature; }

protected:
    double m_density;
    Coord m_velocity;
    double m_pressure;
    double m_temperature;
};