#pragma once
#include "../particle/Particle.h"
#include "../meshes/Element.h"
#include "../variables/Phase.h"
#include "../random/Random.h"
#include "../Param.h"
#include <memory>
#include <vector>

enum class AMRlevel{
    Lv1,
    Lv2,
    Lv3
};

class Cell
{
public:
    using Coord = Particle::Coord;
    Cell() = default;
    virtual ~Cell();

    // Accessors
    const Coord& getposition() const;
    const std::vector<Particle* >& getparticles() const;
    const Phase* getphase() const;
    Element* getelement() const;
    const Coord& getindex() const;
    const std::vector<std::shared_ptr<Cell>>& getchildren() const;
    const int getCollisionNum();
    const int& getparticleStartIndex();
    const double& getdt();
    bool ifcut();
    const double& getmfp();
    const AMRlevel& getAMRlevel();
    // Modifiers
    void setposition(const Coord& position);
    void setelement(Element* element);
    void setindex(const Coord& index);
    void setAMRlevel(const AMRlevel& level);
    void setparticleStartIndex(const int& id);

    // Functions
    void allocatevar();
    void insertparticle(Particle* particle);
    void removeparticle(Particle*  particle);
    void removeallparticles();
    void VTS(); // variable time step algorithm
    void comtimetokenleaving(Particle* particle);
    void clearChildern();
    void genAMRmesh();
    void sortParticle2children();
    void insertchildern(std::shared_ptr<Cell> child);
    double findMaxmfpOverAllchild();
    double findMinmfpOverL2child(std::shared_ptr<Cell> childcell, std::vector<std::shared_ptr<Cell>>& oldchildcells);

    // collision
    virtual void collision();
    // sample micro physics
    virtual void sample();
protected:
    Coord m_position;
    Coord m_index;
    std::vector<Particle*> m_particles;
    std::shared_ptr<Phase> m_phase;
    Element* m_element;
    size_t N_particles {};
    int m_particleStartIdx {}; // index into global particle array
    size_t N_collision {};
    double Mcand_r {}; // the resident caused buy turn double to int

    // add funtion according to "Development of parallel direct simulation Monte Carlo method using a cut-cell Cartesian grid on a single graphics processor"
    double m_mfp {}; // local mean free path
    double m_mps {}; // most probable speed
    double m_dt {tau}; // local time step
    size_t m_weight {Fn}; // particle weight


    // AMR part
    std::vector<std::shared_ptr<Cell>> m_children;
    AMRlevel m_level {AMRlevel::Lv1};
};