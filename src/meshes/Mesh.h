#pragma once
#include "../cell/Cell.h"
#include "object/Geom.h"

class Mesh
{
public:
    Mesh() = default;
    virtual ~Mesh() = default;

    virtual void allocateCells(std::vector<Cell>& cells) {};
    virtual void setelement() {};
    virtual void BindCellwithElement(std::vector<Cell>& cells) {};
    virtual void BindElementwithFace() {};
    virtual void cutcell(Geom* geom) {};
    virtual int getIndex(const Particle::Coord& position) {};
    // Modify
    virtual void setnumberCellsX(const int&) {};
    virtual void setnumberCellsY(const int&) {};
    virtual void setnumberCellsZ(const int&) {};
    virtual void setLengthX(const double& ) {};
    virtual void setLengthY(const double& ) {};
    virtual void setLengthZ(const double& ) {};
    // Access
    virtual const int& getnumberCellsX() {};
    virtual const int& getnumberCellsY() {};
    virtual const int& getnumberCellsZ() {};
    virtual const double& getUnidX() {};
    virtual const double& getUnidY() {};
    virtual const double& getUnidZ() {};

    virtual const double& getLengthX() {};
    virtual const double& getLengthY() {};
    virtual const double& getLengthZ() {};
    virtual const std::vector<std::unique_ptr<Element>>& getElements() {};
};