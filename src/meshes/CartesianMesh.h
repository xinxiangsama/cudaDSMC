#pragma once
#include "Mesh.h"

class CartesianMesh : public Mesh
{
public:
    CartesianMesh() = default;
    ~CartesianMesh() = default;
    void allocateCells(std::vector<Cell>& cells) override;
    void setelement() override;
    void BindCellwithElement(std::vector<Cell>& cells);
    void BindElementwithFace();
    virtual void cutcell(Geom* geom);
    virtual int getIndex(const Particle::Coord& position) override;
    // Modifiers
    void setnumberCellsX(const int& N) override;
    void setnumberCellsY(const int& N) override;
    void setnumberCellsZ(const int& N) override;
    void setLengthX(const double& L) override;
    void setLengthY(const double& L) override;
    void setLengthZ(const double& L) override;
    // Accessers
    const int& getnumberCellsX() override;
    const int& getnumberCellsY() override;
    const int& getnumberCellsZ() override;
    const double& getUnidX() override;
    const double& getUnidY() override;
    const double& getUnidZ() override;
    const double& getLengthX() override;
    const double& getLengthY() override;
    const double& getLengthZ() override;
    const std::vector<std::unique_ptr<Element>>& getElements();
protected:
    int m_numberCellsX = 0; // number of cells in the  domain
    int m_numberCellsY = 0; // number of cells in the  domain
    int m_numberCellsZ = 0; // number of cells in the  domain
    double m_LengthX = 0.0; // length of the  domain in x direction
    double m_LengthY = 0.0; // length of the  domain in y direction
    double m_LengthZ = 0.0; // length of the  domain in z direction
    double m_UnidX = 0.0; // length of the cell in x direction
    double m_UnidY = 0.0; // length of the cell in y direction
    double m_UnidZ = 0.0; // length of the cell in z direction
    std::vector<double> m_dXi; // vector of cell lengths in x direction
    std::vector<double> m_dYj; // vector of cell lengths in y direction
    std::vector<double> m_dZk; // vector of cell lengths in z direction

    std::vector<std::unique_ptr<Element>> m_elements; // vector of elements
};