#pragma once
#include "Face.h"
#include <array>
#include <memory>
#include <Eigen/Dense>
#include "object/Segment.h"
#include "Vertice.h"

class Element 
{
public:
    using Coord = Vertice::Coord;
    Element() = default;
    virtual ~Element() = default;

    // Accessors
    const Coord& getposition() const;
    const double& getL1() const;
    const double& getL2() const;
    const double& getL3() const;
    const double& getvolume() const;
    const std::array<std::unique_ptr<Face>, 6>& getfaces() const;
    Face* getface(size_t index) const;
    const std::vector<std::unique_ptr<Segment>>& getsegments() const;
    const std::vector<std::unique_ptr<LargrangianPoint>>& getIntersectionPs() const;
    std::array<std::unique_ptr<Vertice>, 4>& getvertices();
    std::vector<std::shared_ptr<Element>>& getchildren();
    bool ifcut();
    // Note: The index should be in the range [0, 3] for a 2D element
    // Modifiers
    void setposition(const Coord& position);
    void setL1(const double& L1);
    void setL2(const double& L2);
    void setL3(const double& L3);
    void setvolume(const double& volume);
    void setface(size_t index, std::unique_ptr<Face>&& face);
    void setface(size_t index, std::unique_ptr<Face>& face);
    void insertsegment(std::unique_ptr<Segment>& segment);
    void insertIntersectionP(Eigen::Vector2d& P);
    bool ifContain2d(const Eigen::Vector2d& P);
    bool ifContain(const Eigen::Vector3d& P);

    // AMR
    void genAMRmesh(const int& Nx, const int& Ny, const double& Lx, const double& Ly);
    bool isIntersecting(const Element* other) const;
protected:
    Coord m_position; // Position of the element
    double m_L1; // Length in the first dimension
    double m_L2; // Length in the second dimension
    double m_L3; // Length in the third dimension (for 3D elements)
    double m_volume;  // Volume of the element
    std::array<std::unique_ptr<Face>, 6> m_faces; // Array of faces associated with the element, order is left, right, bottom, top
    // Note: The size of the array should match the number of faces for the element type
    // For example, a 2D quadrilateral element has 4 faces
    std::array<std::unique_ptr<Vertice>, 4> m_vertices;
    std::vector<std::unique_ptr<Segment>> m_segments;
    std::vector<std::unique_ptr<LargrangianPoint>> m_intersectionPs;

    // AMR part
    std::vector<std::shared_ptr<Element>> m_children;
};