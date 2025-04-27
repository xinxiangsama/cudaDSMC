#include "Circle.h"

Circle::Circle(const int &num, const LargrangianPoint::Coord &center, const double &radius)
{
    numLagrangianPoints = num;
    m_center = center;
    m_radius = radius;
}

void Circle::Initialize()
{
    m_points.resize(numLagrangianPoints);
    m_segments.resize(numLagrangianPoints);
    // Set the position of each Lagrangian point
    for (size_t i = 0; i < numLagrangianPoints; ++i)
    {
        double angle = 2.0 * M_PI * i / numLagrangianPoints;
        m_points[i].setPosition(m_center + m_radius * Eigen::Vector3d(cos(angle), sin(angle), 0));
    }

    // Set the segments between Lagrangian points
    for (size_t i = 0; i < numLagrangianPoints; ++i)
    {
        m_segments[i].setleftpoint(&m_points[i]);
        m_segments[i].setrightpoint(&m_points[(i + 1) % numLagrangianPoints]);
        m_points[i].setLeftSegment(&m_segments[(i - 1 + numLagrangianPoints)%numLagrangianPoints]);
        m_points[i].setRightSegment(&m_segments[i]);
    }

    // Set the information of each segment
    for(auto& segment : m_segments)
    {
        auto vector = segment.getrightpoint()->getPosition() - segment.getleftpoint()->getPosition();
        segment.setlength(vector.norm());
        segment.setsloop(vector.normalized());
        segment.setnormal(Eigen::Vector3d(vector.y(), -vector.x(), 0).normalized());
    }
}
