#include "Geom.h"
#include <iostream>
#include "Param.h"
#include <limits>
#include <unordered_map>

void Geom::SortSegment2Element(Element *element)
{
    auto& vertices = element->getvertices();

    // distinguish wall or fluid vertice
    for(auto& vertice : vertices){
        auto p = vertice->getPosition();
        if(cn_PnPolyXpositive({p.x(), p.y()}) && cn_PnPolyXnegetive({p.x(), p.y()}) && cn_PnPolyYpositive({p.x(), p.y()}) && cn_PnPolyYnegetive({p.x(), p.y()})){
            vertice->setWall();
        }
    }
    // compute intersection point in cell dege
    for(auto& vertice : vertices){
        if (vertice->isWall()) {
            auto p = Eigen::Vector2d({vertice->getPosition()[0], vertice->getPosition()[1]});
            Eigen::Vector2d inter_xp = PnPolyXpositiveIntersection(p);
            Eigen::Vector2d inter_xn = PnPolyXnegetiveIntersection(p);
            Eigen::Vector2d inter_yp = PnPolyYpositiveIntersection(p);
            Eigen::Vector2d inter_yn = PnPolyYnegetiveIntersection(p);

            if(element->ifContain2d(inter_xp))
                element->insertIntersectionP(inter_xp);

            if(element->ifContain2d(inter_xn))
                element->insertIntersectionP(inter_xn);

            if(element->ifContain2d(inter_yp))
                element->insertIntersectionP(inter_yp);

            if(element->ifContain2d(inter_yn))
                element->insertIntersectionP(inter_yn);
        }
        
    }
    if(element->getIntersectionPs().size()){
        if(element->getIntersectionPs().size() != 2){
            std::cerr <<"the intersection point num is not as imaged 2 And it's "<<element->getIntersectionPs().size()<<std::endl;
        }else{
            double esp = 1.0e-3;
            auto& pts = element->getIntersectionPs();
            Eigen::Vector2d p0(Eigen::Vector2d{pts[0]->getPosition().x(), pts[0]->getPosition().y()});
            Eigen::Vector2d p1(Eigen::Vector2d{pts[1]->getPosition().x(), pts[1]->getPosition().y()});

            Eigen::Vector2d normal = (p1 - p0).unitOrthogonal();  // 顺时针旋转90°的单位法向
            Eigen::Vector2d midpoint = 0.5 * (p0 + p1) + esp * normal;
        
            // 判断中点是否在 solid 内部
            bool isMidInSolid = cn_PnPolyXpositive(midpoint) &&
                                cn_PnPolyXnegetive(midpoint) &&
                                cn_PnPolyYpositive(midpoint) &&
                                cn_PnPolyYnegetive(midpoint);
        
            if (isMidInSolid) {
                // 法向应指向 fluid，当前法向朝 solid -> 取反
                normal = -normal;
            }
        // std::cout << normal.transpose()<<std::endl;
        auto segment = std::make_unique<Segment>();
        segment->setnormal({normal.x(), normal.y(), 0.0});
        segment->setleftpoint(pts[0].get());
        segment->setrightpoint(pts[1].get());
        element->insertsegment(segment);
        }
    }

}

int Geom::cn_PnPolyXpositive(const Eigen::Vector2d &P)
{
    int cn = 0;
    // loop through all edges of polygon
    for(int i = 0; i < numLagrangianPoints; ++i){
        auto index1 = i;
        auto index2 = (i + 1) % numLagrangianPoints;
        auto point1 = m_points[index1];
        auto point2 = m_points[index2];
        auto V1 = Eigen::Vector2d{point1.getPosition()[0], point1.getPosition()[1]};
        auto V2 = Eigen::Vector2d{point2.getPosition()[0], point2.getPosition()[1]};

        if(((V1.y() <= P.y()) && (V2.y() > P.y())) // an upward crossing
            ||((V1.y() > P.y()) && (V2.y() <= P.y()))){ //an downward crossing
                double vt = static_cast<double>((P.y() - V1.y()) / (V2.y() - V1.y()));
                if (std::abs(V2.y() - V1.y()) < 1e-12) {
                    continue; // Avoid division by zero
                    std::cerr <<"infinity"<<std::endl;
                }
                double ixc = V1.x() + vt * (V2.x() - V1.x());
                if(P.x() < ixc){
                    cn = !cn;
                }
            }
    }

    return cn; // 0 if even (out), and 1 if odd (in)
}

int Geom::cn_PnPolyYpositive(const Eigen::Vector2d &P)
{
    int cn = 0;
    for (int i = 0; i < numLagrangianPoints; ++i) {
        int index1 = i;
        int index2 = (i + 1) % numLagrangianPoints;

        auto point1 = m_points[index1];
        auto point2 = m_points[index2];
        Eigen::Vector2d V1(point1.getPosition()[0], point1.getPosition()[1]);
        Eigen::Vector2d V2(point2.getPosition()[0], point2.getPosition()[1]);

        if (((V1.x() <= P.x()) && (V2.x() > P.x())) || ((V1.x() > P.x()) && (V2.x() <= P.x()))) {
            double vt = (P.x() - V1.x()) / (V2.x() - V1.x());
            if (std::abs(V2.x() - V1.x()) < 1e-12) {
                continue; // Avoid division by zero
                std::cerr <<"infinity"<<std::endl;
            }
            double iyc = V1.y() + vt * (V2.y() - V1.y());

            if (P.y() < iyc) {
                cn = !cn;
            }
        }
    }

    return cn;
}


int Geom::cn_PnPolyXnegetive(const Eigen::Vector2d &P)
{
    int cn = 0;
    // loop through all edges of polygon

    for(int i = 0; i < numLagrangianPoints; ++i){
        auto index1 = i;
        auto index2 = (i + 1) % numLagrangianPoints;
        auto point1 = m_points[index1];
        auto point2 = m_points[index2];
        auto V1 = Eigen::Vector2d{point1.getPosition()[0], point1.getPosition()[1]};
        auto V2 = Eigen::Vector2d{point2.getPosition()[0], point2.getPosition()[1]};

        if(((V1.y() <= P.y()) && (V2.y() > P.y())) // an upward crossing
            ||((V1.y() > P.y()) && (V2.y() <= P.y()))){ //an downward crossing
                double vt = static_cast<double>((P.y() - V1.y()) / (V2.y() - V1.y()));
                if (std::abs(V2.y() - V1.y()) < 1e-12) {
                    continue; // Avoid division by zero
                    std::cerr <<"infinity"<<std::endl;
                }
                double ixc = V1.x() + vt * (V2.x() - V1.x());
                if(P.x() > ixc){
                    cn = !cn;
                }
            }
    }

    return cn;
}

int Geom::cn_PnPolyYnegetive(const Eigen::Vector2d &P)
{   
    int cn = 0;
    for (int i = 0; i < numLagrangianPoints; ++i) {
        int index1 = i;
        int index2 = (i + 1) % numLagrangianPoints;

        auto point1 = m_points[index1];
        auto point2 = m_points[index2];
        Eigen::Vector2d V1(point1.getPosition()[0], point1.getPosition()[1]);
        Eigen::Vector2d V2(point2.getPosition()[0], point2.getPosition()[1]);

        if (((V1.x() <= P.x()) && (V2.x() > P.x())) || ((V1.x() > P.x()) && (V2.x() <= P.x()))) {
            double vt = (P.x() - V1.x()) / (V2.x() - V1.x());
            if (std::abs(V2.x() - V1.x()) < 1e-12) {
                continue; // Avoid division by zero
                std::cerr <<"infinity"<<std::endl;
            }
            double iyc = V1.y() + vt * (V2.y() - V1.y());

            if (P.y() > iyc) {
                cn = !cn;
            }
        }
    }

    return cn;
}


// +x方向
Eigen::Vector2d Geom::PnPolyXpositiveIntersection(const Eigen::Vector2d &P)
{
    for (int i = 0; i < numLagrangianPoints; ++i) {
        auto V1 = m_points[i].getPosition();
        auto V2 = m_points[(i + 1) % numLagrangianPoints].getPosition();

        if (((V1.y() <= P.y()) && (V2.y() > P.y())) ||
            ((V1.y() > P.y()) && (V2.y() <= P.y()))) {

            double dy = V2.y() - V1.y();
            if (std::abs(dy) < 1e-12) continue;

            double vt = (P.y() - V1.y()) / dy;
            double ixc = V1.x() + vt * (V2.x() - V1.x());

            if (ixc > P.x()) {
                return Eigen::Vector2d(ixc, P.y());
            }
        }
    }
    return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
}

// -x方向
Eigen::Vector2d Geom::PnPolyXnegetiveIntersection(const Eigen::Vector2d &P)
{
    for (int i = 0; i < numLagrangianPoints; ++i) {
        auto V1 = m_points[i].getPosition();
        auto V2 = m_points[(i + 1) % numLagrangianPoints].getPosition();

        if (((V1.y() <= P.y()) && (V2.y() > P.y())) ||
            ((V1.y() > P.y()) && (V2.y() <= P.y()))) {

            double dy = V2.y() - V1.y();
            if (std::abs(dy) < 1e-12) continue;

            double vt = (P.y() - V1.y()) / dy;
            double ixc = V1.x() + vt * (V2.x() - V1.x());

            if (ixc < P.x()) {
                return Eigen::Vector2d(ixc, P.y());
            }
        }
    }
    return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
}

// +y方向
Eigen::Vector2d Geom::PnPolyYpositiveIntersection(const Eigen::Vector2d &P)
{
    for (int i = 0; i < numLagrangianPoints; ++i) {
        auto V1 = m_points[i].getPosition();
        auto V2 = m_points[(i + 1) % numLagrangianPoints].getPosition();

        if (((V1.x() <= P.x()) && (V2.x() > P.x())) ||
            ((V1.x() > P.x()) && (V2.x() <= P.x()))) {

            double dx = V2.x() - V1.x();
            if (std::abs(dx) < 1e-12) continue;

            double vt = (P.x() - V1.x()) / dx;
            double iyc = V1.y() + vt * (V2.y() - V1.y());

            if (iyc > P.y()) {
                return Eigen::Vector2d(P.x(), iyc);
            }
        }
    }
    return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
}

// -y方向
Eigen::Vector2d Geom::PnPolyYnegetiveIntersection(const Eigen::Vector2d &P)
{
    for (int i = 0; i < numLagrangianPoints; ++i) {
        auto V1 = m_points[i].getPosition();
        auto V2 = m_points[(i + 1) % numLagrangianPoints].getPosition();

        if (((V1.x() <= P.x()) && (V2.x() > P.x())) ||
            ((V1.x() > P.x()) && (V2.x() <= P.x()))) {

            double dx = V2.x() - V1.x();
            if (std::abs(dx) < 1e-12) continue;

            double vt = (P.x() - V1.x()) / dx;
            double iyc = V1.y() + vt * (V2.y() - V1.y());

            if (iyc < P.y()) {
                return Eigen::Vector2d(P.x(), iyc);
            }
        }
    }
    return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
}
