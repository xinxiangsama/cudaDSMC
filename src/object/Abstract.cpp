#include "Abstract.h"

void Abstract::Initialize()
{
    // numLagrangianPoints = 4;
    // m_points.resize(4);
    // m_segments.resize(4);

    // double offset = 0.0725;
    // m_points[0].setPosition({3e-1 + offset, 5e-1, 0.0});
    // m_points[1].setPosition({3e-1 + offset, 3e-1, 0.0});
    // m_points[2].setPosition({5e-1 + offset, 2e-1, 0.0});
    // m_points[3].setPosition({5e-1 + offset, 6e-1, 0.0});


    /*===========NACA0012==================*/
    // NACA0012 airfoil(200, 0.5, Eigen::Vector3d(0.2255, 0.4255, 0.0));
    // auto nacaPoints = airfoil.generatePoints();
    // numLagrangianPoints = nacaPoints.size();
    // m_points.resize(nacaPoints.size());
    // m_segments.resize(nacaPoints.size());
    // // 然后使用 setPosition() 存储这些点到你的 LagrangianPoint 对象中
    // for (int i = 0; i < nacaPoints.size(); ++i) {
    //     m_points[i].setPosition(nacaPoints[i]);
    //     std::cout << nacaPoints[i].transpose()<<std::endl;
    // }

    /*================阿波罗返回舱===================*/
    // numLagrangianPoints = 6;
    // m_points.resize(numLagrangianPoints);
    // m_points[0].setPosition({0.0, 0.0, 0.0});
    // m_points[1].setPosition({0.3743, 1.8368, 0.0});
    // m_points[2].setPosition({0.5543, 1.9558, 0.0});
    // m_points[3].setPosition({0.6608, 1.9242, 0.0});
    // m_points[4].setPosition({3.3254, 0.1938, 0.0});
    // m_points[5].setPosition({3.3406, 0.0, 0.0});
    // auto offset {Eigen::Vector3d(0.2255, 0.4255, 0.0)};
    // double scale {0.1};
    // for(auto& point : m_points){
    //     point.setPosition(point.getPosition() * scale + offset);
    // }

    /*===========25/55 degree biconic==============*/
    // numLagrangianPoints = 7;
    // m_points.resize(numLagrangianPoints);
    // m_points[0].setPosition({0.0, 0.0, 0.0});
    // m_points[1].setPosition({0.8344376, 0.3891046, 0.0});
    // m_points[2].setPosition({1.8286591, 1.809, 0.0});
    // m_points[3].setPosition({1.937, 1.809, 0.0});
    // m_points[4].setPosition({1.937, -1.809, 0.0});
    // m_points[5].setPosition({1.8286591, -1.809, 0.0});
    // m_points[6].setPosition({0.8344376, -0.3891046, 0.0});
    // auto offset {Eigen::Vector3d(0.4255, 1.9255, 0.0)};
    // for(auto& point : m_points){
    //     point.setPosition(point.getPosition() + offset);
    // }

    /*============= flat plate==================*/
    // numLagrangianPoints = 4;
    // m_points.resize(numLagrangianPoints);
    // m_points[0].setPosition({0.1000, 0.005, 0.0});
    // m_points[1].setPosition({0.9000, 0.005, 0.0});
    // m_points[2].setPosition({0.9000, -0.005, 0.0});
    // m_points[3].setPosition({0.1000, -0.005, 0.0});
    // auto offset {Eigen::Vector3d(0.05255, 0.4955, 0.0)};
    // for(auto& point : m_points){
    //     point.setPosition(point.getPosition() + offset);
    // }

    /*===========Bluet==============*/
    // numLagrangianPoints = 4;
    // m_points.resize(numLagrangianPoints);
    // m_points[0].setPosition({0.01928, 0.01258, 0.0});
    // m_points[1].setPosition({0.3556, 0.0762, 0.0});
    // m_points[2].setPosition({0.3556, -0.0762, 0.0});
    // m_points[3].setPosition({0.01928, -0.01258, 0.0});


    // auto offset {Eigen::Vector3d(0.15255, 0.4955, 0.0)};
    // for(auto& point : m_points){
    //     point.setPosition(point.getPosition() + offset);
    // }

    /*===========二维扩张管道=========*/
    numLagrangianPoints = 8;
    m_points.resize(numLagrangianPoints);
    m_points[0].setPosition({0.0, 0.0, 0.0});
    m_points[1].setPosition({101.7, 0.0, 0.0});
    m_points[2].setPosition({145, 25, 0.0});
    m_points[3].setPosition({170, 25, 0.0});
    m_points[4].setPosition({170, 0.0, 0.0});
    m_points[5].setPosition({145, 0.0, 0.0});
    m_points[6].setPosition({101.7, -10.0, 0.0});
    m_points[7].setPosition({37.32, -10, 0.0});

    auto offset {Eigen::Vector3d(0.03555, 0.04255, 0.0)};
    double scale {1e-3};
    for(auto& point : m_points){
        point.setPosition(point.getPosition() * scale + offset);
    }
}

void Abstract::ReadFromTxt()
{

}