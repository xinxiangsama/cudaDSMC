#pragma once
#include <Eigen/Dense>

class Face
{
public:
    using Coord = Eigen::Vector3d;
    Face() = default;
    Face(const Coord& position, const Coord& normal)
        : m_position(position), m_normal(normal) {}
    virtual ~Face() = default;

    // Accessors
    const Coord& getposition() const { return m_position; }
    const Coord& getnormal() const { return m_normal; }

    // Modifiers
    void setposition(const Coord& position) { m_position = position; }
    void setnormal(const Coord& normal) { m_normal = normal; }
    // Functions
protected:
    Coord m_position; // Position of the face
    Coord m_normal;   // Normal vector of the face
};