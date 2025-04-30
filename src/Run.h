#pragma once
#include <mpi.h>
#include <memory>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "meshes/CartesianMesh.h"
#include "boundary/WallBoundary.h"
#include "boundary/OutletBoundary.h"
#include "boundary/PeriodicBoundary.h"
#include "boundary/InletBoundary.h"
#include "boundary/Boundary.cuh"
#include "io/Output.h"
#include "cell/Cell.h"
#include "cell/Cell.cuh"
#include "object/Circle.h"
#include "object/Square.h"
#include "particle/Particle.cuh"

class Output;
class Run
{
public:
    Run() = default;
    ~Run() = default;

    /*============Initialize==========*/
    void initialize(int argc, char** argv);
    /*============Main calculate loop===========*/
    void solver();
    /*============Finalize===========*/
    void finalize();

    void assignParticle(const double& coef);
    void TransferParticlesFromHostToDevice();
    void TransferConstants();
    void TransferCellsFromHostToDevice();
    void TransferCellsFromDeviceToHost();
    void particlemove();
    void ressignParticle();
    void collision();

protected:
    std::vector<Cell> m_cells;
    std::vector<Particle> m_particles;
    std::shared_ptr<GPUParticles> d_particles;
    std::shared_ptr<GPUCells> d_cells;
    GPUBoundary::Boundary* d_boundaries;
    size_t numparticlelocal;

    /*mesh*/
    std::unique_ptr<Mesh> m_mesh;
    std::unique_ptr<Geom> m_geom;
    std::unique_ptr<Boundary> inlet;
    std::unique_ptr<Boundary> outlet;
    std::unique_ptr<Boundary> wall1; // bottom wall
    std::unique_ptr<Boundary> wall2; // top wall
    std::unique_ptr<Boundary> wall3; // front wall
    std::unique_ptr<Boundary> wall4; // back wall

    /*output*/
    std::unique_ptr<Output> m_output;

    friend class Output;
};
