#pragma once
#include "Segment.h"
#include "../particle/Particle.h"
#include "meshes/Element.h"

class Geom
{
public:
    Geom() = default;
    Geom(const size_t& num) : numLagrangianPoints(num) {};
    virtual ~Geom() = default;

    virtual void Initialize() = 0;

    void SortSegment2Element(Element* element);
    int cn_PnPolyXpositive(const Eigen::Vector2d& P);
    int cn_PnPolyYpositive(const Eigen::Vector2d& P);
    int cn_PnPolyXnegetive(const Eigen::Vector2d& P);
    int cn_PnPolyYnegetive(const Eigen::Vector2d& P);

    Eigen::Vector2d PnPolyXpositiveIntersection(const Eigen::Vector2d& P);
    Eigen::Vector2d PnPolyYpositiveIntersection(const Eigen::Vector2d& P);
    Eigen::Vector2d PnPolyXnegetiveIntersection(const Eigen::Vector2d& P);
    Eigen::Vector2d PnPolyYnegetiveIntersection(const Eigen::Vector2d& P);

protected:
    size_t numLagrangianPoints = 0;
    std::vector<LargrangianPoint> m_points;
    std::vector<Segment> m_segments;
};

// Geom.h
enum class RayDirection {
    X_POS,  // +x direction
    X_NEG,  // -x direction
    Y_POS,  // +y direction
    Y_NEG   // -y direction
};
