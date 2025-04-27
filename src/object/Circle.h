#pragma once
#include "Geom.h"

class Circle : public Geom
{
public:
    Circle() = default;
    Circle(const int& num, const LargrangianPoint::Coord& center, const double& radius);
    virtual ~Circle() = default;

    virtual void Initialize();

protected:
    LargrangianPoint::Coord m_center;
    double m_radius;
};