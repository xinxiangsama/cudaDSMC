#include "Square.h"

Square::Square(const int& numPerSide, const LargrangianPoint::Coord& center, const double& halfLength)
{
    m_numPerSide = numPerSide;
    m_center = center;
    m_halfLength = halfLength;

    numLagrangianPoints = 4 * numPerSide;
}

void Square::Initialize()
{   
    m_points.resize(4);
    m_points[0].setPosition(m_center + Eigen::Vector3d({-m_halfLength, -m_halfLength, 0.0}));
    m_points[1].setPosition(m_center + Eigen::Vector3d({m_halfLength, -m_halfLength, 0.0}));
    m_points[2].setPosition(m_center + Eigen::Vector3d({m_halfLength, m_halfLength, 0.0}));
    m_points[3].setPosition(m_center + Eigen::Vector3d({-m_halfLength, m_halfLength, 0.0}));

    // m_points.resize(numLagrangianPoints);
    // m_segments.resize(numLagrangianPoints);

    // // 计算每条边的点间距
    // double edgeSpacing = 2.0 * m_halfLength / (m_numPerSide - 1);

    // // 起始点：左下角
    // int idx = 0;

    // // Bottom edge (left to right)
    // for (int i = 0; i < m_numPerSide; ++i)
    // {
    //     double x = -m_halfLength + i * edgeSpacing;
    //     m_points[idx++].setPosition(m_center + Eigen::Vector3d(x, -m_halfLength, 0));
    // }

    // // Right edge (bottom to top)
    // for (int i = 1; i < m_numPerSide; ++i)
    // {
    //     double y = -m_halfLength + i * edgeSpacing;
    //     m_points[idx++].setPosition(m_center + Eigen::Vector3d(m_halfLength, y, 0));
    // }

    // // Top edge (right to left)
    // for (int i = 1; i < m_numPerSide; ++i)
    // {
    //     double x = m_halfLength - i * edgeSpacing;
    //     m_points[idx++].setPosition(m_center + Eigen::Vector3d(x, m_halfLength, 0));
    // }

    // // Left edge (top to bottom)
    // for (int i = 1; i < m_numPerSide - 1; ++i)
    // {
    //     double y = m_halfLength - i * edgeSpacing;
    //     m_points[idx++].setPosition(m_center + Eigen::Vector3d(-m_halfLength, y, 0));
    // }

    // // 设置 segments 与点之间的连接关系
    // for (int i = 0; i < numLagrangianPoints; ++i)
    // {
    //     m_segments[i].setleftpoint(&m_points[i]);
    //     m_segments[i].setrightpoint(&m_points[(i + 1) % numLagrangianPoints]);

    //     m_points[i].setLeftSegment(&m_segments[(i - 1 + numLagrangianPoints) % numLagrangianPoints]);
    //     m_points[i].setRightSegment(&m_segments[i]);
    // }

    // // 设置每个 segment 的几何属性
    // for (auto& segment : m_segments)
    // {
    //     auto vector = segment.getrightpoint()->getPosition() - segment.getleftpoint()->getPosition();
    //     segment.setlength(vector.norm());
    //     segment.setsloop(vector.normalized());
    //     segment.setnormal(Eigen::Vector3d(vector.y(), -vector.x(), 0).normalized()); // 逆时针法向
    // }
}
