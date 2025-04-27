#pragma once
#include "Geom.h"

class Square : public Geom
{
public:
    Square() = default;
    Square(const int& numPerSide, const LargrangianPoint::Coord& center, const double& halfLength);
    virtual ~Square() = default;

    virtual void Initialize();

protected:
    LargrangianPoint::Coord m_center;
    double m_halfLength;
    int m_numPerSide; // 拉格朗日点在每条边上的数量
};
