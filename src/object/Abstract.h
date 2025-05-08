#pragma once 
#include "Geom.h"
#include "Bezier.h"
#include <vector>
#include <cmath>
#include <Eigen/Dense>

class Abstract : public Geom
{   
    void ReadFromTxt();
    virtual void Initialize() override;
};



class NACA0012 {
public:
    NACA0012(int numPoints = 100, double chordLength = 1.0, Eigen::Vector3d offset = Eigen::Vector3d::Zero())
        : m_numPoints(numPoints), m_chord(chordLength), m_offset(offset) {}

    std::vector<Eigen::Vector3d> generatePoints() const {
        std::vector<Eigen::Vector3d> points;
        points.reserve(2 * m_numPoints - 1);  // 去掉一个重复点后的容量

        // 下表面：从尾部到头部（排除头部点 i == 0）
        for (int i = m_numPoints - 1; i > 0; --i) {
            double x = static_cast<double>(i) / (m_numPoints - 1) * m_chord;
            double yt = thickness(x);
            points.emplace_back(x, -yt, 0.0);
        }

        // 上表面：从头部到尾部（保留头部点）
        for (int i = 0; i < m_numPoints; ++i) {
            double x = static_cast<double>(i) / (m_numPoints - 1) * m_chord;
            double yt = thickness(x);
            points.emplace_back(x, yt, 0.0);
        }

        // 应用位置偏移
        for (auto& p : points) {
            p += m_offset;
        }

        return points;
    }

private:
    int m_numPoints;
    double m_chord;
    Eigen::Vector3d m_offset;

    double thickness(double x) const {
        double t = 0.12 * m_chord;
        double x_c = x / m_chord;

        return 5 * t * (
                0.2969 * sqrt(x_c)
            - 0.1260 * x_c
            - 0.3516 * pow(x_c, 2)
            + 0.2843 * pow(x_c, 3)
            - 0.1015 * pow(x_c, 4)
        );
    }
};

