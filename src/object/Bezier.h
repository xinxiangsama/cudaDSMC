#pragma once 
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

class BezierCurve {
public:
    BezierCurve(const std::vector<Eigen::Vector3d>& controlPoints)
        : m_controlPoints(controlPoints) {}

    // 计算 Bézier 曲线上的点
    Eigen::Vector3d getPointOnCurve(double t) const {
        int n = m_controlPoints.size() - 1;
        Eigen::Vector3d point(0.0, 0.0, 0.0);  // 使用三维点

        for (int i = 0; i <= n; ++i) {
            double binomialCoeff = binomial(n, i);
            double b = binomialCoeff * pow(1 - t, n - i) * pow(t, i);
            point += b * m_controlPoints[i];
        }

        return point;
    }

    // 离散化曲线
    void discretize(int numPoints) {
        m_points.clear();
        for (int i = 0; i < numPoints; ++i) {
            double t = static_cast<double>(i) / (numPoints - 1);  // 从0到1均匀采样
            m_points.push_back(getPointOnCurve(t));
        }
    }

    const std::vector<Eigen::Vector3d>& getPoints() const {
        return m_points;
    }

private:
    std::vector<Eigen::Vector3d> m_controlPoints;  // 存储三维控制点
    std::vector<Eigen::Vector3d> m_points;  // 存储离散化后的三维曲线点

    // 计算组合数 C(n, k)
    int binomial(int n, int k) const {
        if (k == 0 || k == n) return 1;
        return binomial(n - 1, k - 1) + binomial(n - 1, k);
    }
};