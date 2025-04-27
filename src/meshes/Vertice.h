#pragma once
#include <Eigen/Dense>
enum struct VerticeType{
    Wall,
    Fluid
};
class Vertice
{
public:
    using Coord = Eigen::Vector3d;
    Vertice() = default;
    Vertice(const Coord& position): m_position(position) {};
    ~Vertice() = default;

    void setWall() {m_type = VerticeType::Wall;}
    void setPosition(const Coord& position) {m_position = position;}

    bool isWall() {return m_type == VerticeType::Wall;}
    const Coord& getPosition() const {return m_position;}
protected:
    Coord m_position;
    VerticeType m_type {VerticeType::Fluid};
};