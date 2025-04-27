#include "Run.h"
#include <chrono>
#include <iomanip>
double Vstd;
double Vmax;
double VHS_coe;
std::unique_ptr<Random> randomgenerator;

auto GammaFun = [](double xlen){
    double A, ylen, GAM;
    
    A = 1.0;
    ylen = xlen;
    
    if (ylen < 1.0) {
        A = A / ylen;
    } else {
        while (ylen >= 1.0) {
            ylen = ylen - 1;
            A = A * ylen;
        }
    }
    
    GAM = A * (1.0 - 0.5748 * ylen + 0.9512 * ylen * ylen - 0.6998 * ylen * ylen * ylen + 0.4245 * ylen * ylen * ylen * ylen - 0.1010 * ylen * ylen * ylen * ylen * ylen);
    
    return GAM;
};

void Run::initialize(int argc, char **argv)
{   
    /*cal base var*/
    Vstd = sqrt(2 * boltz * T / mass);
    Vmax = 2 * sqrt(8 / M_PI) * Vstd;
    VHS_coe = GammaFun(2.5 - Vtl);
    /*mesh part*/
    m_mesh = std::make_unique<CartesianMesh>();
    m_mesh->setLengthX(L1);
    m_mesh->setLengthY(L2);
    m_mesh->setLengthZ(L3);
    m_mesh->setnumberCellsX(N1);
    m_mesh->setnumberCellsY(N2);
    m_mesh->setnumberCellsZ(N3);

    m_geom = std::make_unique<Circle>(128, LargrangianPoint::Coord{Center_x, Center_y, 0.0}, Radius);
    // m_geom = std::make_unique<Square>(4, LargrangianPoint::Coord{Center_x, Center_y, 0.0}, Radius);
    m_geom->Initialize();


    /*second mesh part*/
    m_mesh->allocateCells(m_cells);
    m_mesh->setelement();
    m_mesh->BindElementwithFace();
    m_mesh->BindCellwithElement(m_cells);
    m_mesh->cutcell(m_geom.get());
    /*boundary part*/
    // 有6个边界
    GPUBoundary::Boundary h_boundaries[6];
    //  (x=0，periodic))
    h_boundaries[0].point = make_double3(0.0, 0.5 * L2, 0.5 * L3);
    h_boundaries[0].normal = make_double3(1.0, 0.0, 0.0);
    h_boundaries[0].type = GPUBoundary::BoundaryType::PERIODIC;

    //  (x=L1，periodic))
    h_boundaries[1].point = make_double3(L1, 0.5 * L2, 0.5 * L3);
    h_boundaries[1].normal = make_double3(-1.0, 0.0, 0.0);
    h_boundaries[1].type = GPUBoundary::BoundaryType::PERIODIC;

    // (y=L2，periodic)
    h_boundaries[2].point = make_double3(0.5 * L1, L2, 0.5 * L3);
    h_boundaries[2].normal = make_double3(0.0, -1.0, 0.0);
    h_boundaries[2].type = GPUBoundary::BoundaryType::PERIODIC;

    // (y=0，periodic)
    h_boundaries[3].point = make_double3(0.5 * L1, 0.0, 0.5 * L3);
    h_boundaries[3].normal = make_double3(0.0, 1.0, 0.0);
    h_boundaries[3].type = GPUBoundary::BoundaryType::PERIODIC;

    // (z=0，periodic)
    h_boundaries[4].point = make_double3(0.5 * L1, 0.5 * L2, 0.0);
    h_boundaries[4].normal = make_double3(0.0, 0.0, 1.0);
    h_boundaries[4].type = GPUBoundary::BoundaryType::PERIODIC;

    // (z=L3，periodic)
    h_boundaries[5].point = make_double3(0.5 * L1, 0.5 * L2, L3);
    h_boundaries[5].normal = make_double3(0.0, 0.0, -1.0);
    h_boundaries[5].type = GPUBoundary::BoundaryType::PERIODIC;
    // malloc并拷贝到GPU
    Boundary* d_boundaries;
    cudaMalloc(&d_boundaries, sizeof(Boundary) * 6);
    cudaMemcpy(d_boundaries, h_boundaries, sizeof(Boundary) * 6, cudaMemcpyHostToDevice);

    /*random part*/
    randomgenerator = std::make_unique<Random>();
    /*output part*/
    m_output = std::make_unique<Output>(this);
    /*Initial particle phase*/
    assignParticle(1.0);
    for(auto& cell : m_cells){
        cell.allocatevar();
    }
    d_particles = std::make_shared<GPUParticles>(m_particles.size());
    d_cells = std::make_shared<GPUCells>(m_cells.size());
    TransferParticlesFromHostToDevice();
    TransferCellsFromHostToDevice();
}

void Run::assignParticle(const double& coef)
{
    numparticlelocal = static_cast<int>(N_Particle * coef);
    m_particles.reserve(numparticlelocal);
    if(N_Particle % (N1 * N2 * N3) != 0){
        std::cerr <<"particle can't be devided by mesh" << std::endl;
    }
    std::random_device rd;
    auto numparticlepercell = numparticlelocal / (m_mesh->getnumberCellsX() * m_mesh->getnumberCellsY() * m_mesh->getnumberCellsZ());
    int particle_global_id {};
    for(auto& cell: m_cells){
        std::mt19937 gen(rd() + cell.getindex()[0] + cell.getindex()[1] + cell.getindex()[2]);
        int particle_local_id{};
        for(int i = 0; i < numparticlepercell; ++i){
            auto rx = randomgenerator->getrandom01();
            auto ry = randomgenerator->getrandom01();
            auto rz = randomgenerator->getrandom01();

            double x = cell.getposition()(0) + (rx - 0.5) * m_mesh->getUnidX();
            double y = cell.getposition()(1) + (ry - 0.5) * m_mesh->getUnidY();
            double z = cell.getposition()(2) + (rz - 0.5) * m_mesh->getUnidZ();
            // if((x - Center_x) * (x - Center_x) + (y - Center_y) * (y - Center_y) <= 1.5*(Radius * Radius)){
            //     continue;
            // }
            auto velocity = randomgenerator->MaxwellDistribution(Vstd);
            velocity(0) += V_jet;
            m_particles.emplace_back(mass, Eigen::Vector3d{x, y, z}, velocity);
            std::prev(m_particles.end())->setcellID(m_mesh->getIndex(Eigen::Vector3d{x, y, z}));
            std::prev(m_particles.end())->setlocalID(particle_local_id);
            std::prev(m_particles.end())->setglobalID(particle_global_id);
            cell.insertparticle(&(*std::prev(m_particles.end())));
            particle_local_id++;
            particle_global_id++;
        }
    }

    //set particle start index for cell
    int startIndex {};
    for(auto& cell : m_cells){
        cell.setparticleStartIndex(startIndex);
        startIndex += cell.getparticles().size();
    }

    // // set particle global index
    // for(auto& particle : m_particles){
    //     auto& cell {m_cells[particle.getcellID()]};
    //     particle.setglobalID(particle.getlocalID() + cell.getparticleStartIndex());
    // }
}

void Run::particlemove()
{   
    d_particles->Move(tau, 128, d_boundaries);
}

void Run::TransferParticlesFromHostToDevice()
{   
    int N = m_particles.size();
    std::vector<double> h_pos_x(N), h_pos_y(N), h_pos_z(N);
    std::vector<double> h_vel_x(N), h_vel_y(N), h_vel_z(N);
    std::vector<double> h_mass(N);
    std::vector<int> h_global_id(N), h_local_id(N), h_cell_id(N);

    for (int i = 0; i < N; ++i) {
        auto& particle = m_particles[i];
        h_mass[i] = particle.getmass();

        h_pos_x[i] = particle.getposition()(0);
        h_pos_y[i] = particle.getposition()(1);
        h_pos_z[i] = particle.getposition()(2);
        h_global_id[i] = particle.getglobalID();
        h_local_id[i] = particle.getlocalID();
        h_cell_id[i] = particle.getcellID();
        h_vel_x[i] = particle.getvelocity()(0);
        h_vel_y[i] = particle.getvelocity()(1);
        h_vel_z[i] = particle.getvelocity()(2);
    }
    d_particles->UploadFromHost(h_mass.data(), 
            h_pos_x.data(), h_pos_y.data(), h_pos_z.data(), 
            h_vel_x.data(), h_vel_y.data(), h_vel_z.data(), h_global_id.data(), h_local_id.data(), h_cell_id.data());
}

void Run::TransferCellsFromHostToDevice()
{
    int N = m_cells.size();
    std::vector<int> h_particleNum(N), h_particleStartIndex(N);
    for(int i = 0; i < N; ++i){
        auto& cell = m_cells[i];
        h_particleNum[i] = cell.getparticles().size();
        h_particleStartIndex[i] = cell.getparticleStartIndex();
    }
    auto h_Unidx {m_mesh->getUnidX()};
    auto h_Unidy {m_mesh->getUnidY()};
    auto h_Unidz {m_mesh->getUnidZ()};
    d_cells->UploadFromHost(h_particleNum.data(), h_particleStartIndex.data(), h_Unidx, h_Unidy, h_Unidz);
}

void Run::TransferCellsFromDeviceToHost()
{
    int N = m_cells.size();
    std::vector<double> h_Temperature(N), h_Rho(N), h_Pressure(N);
    std::vector<double3> h_Velocity(N);
    d_cells->DownloadToHost(h_Temperature.data(), h_Rho.data(), h_Pressure.data(), h_Velocity.data());
    for (int i = 0; i < N; ++i) {
        auto& cell = m_cells[i];
        cell.setTemperature(h_Temperature[i]);
        cell.setRho(h_Rho[i]);
        cell.setPressure(h_Pressure[i]);
        cell.setVelocity(Eigen::Vector3d(h_Velocity[i].x, h_Velocity[i].y, h_Velocity[i].z));
    }
}

void Run::ressignParticle()
{   
    d_cells->CalparticleNum(d_particles->d_pos_x, d_particles->d_pos_y, d_particles->d_pos_z, d_particles->local_id, d_particles->cell_id,d_particles->N, N1, N2, N3, 128);
    d_cells->CalparticleStartIndex();    
    d_particles->Sort();
}

void Run::collision()
{
    d_cells->Collision(d_particles->d_vel_x, d_particles->d_vel_y, d_particles->d_vel_z);
}


void Run::solver()
{
    for(size_t iter = 0; iter < 10000; ++iter){

        auto t_start = std::chrono::high_resolution_clock::now();

        auto t_particlemove_start = std::chrono::high_resolution_clock::now();
        particlemove();
        auto t_particlemove_end = std::chrono::high_resolution_clock::now();

        auto t_ressign_start = std::chrono::high_resolution_clock::now();
        ressignParticle();
        auto t_ressign_end = std::chrono::high_resolution_clock::now();

        auto t_collision_start = std::chrono::high_resolution_clock::now();
        collision();
        auto t_collision_end = std::chrono::high_resolution_clock::now();
    
        auto t_end = std::chrono::high_resolution_clock::now();
        // 输出计时统计
        std::stringstream ss;
        ss << "========================================\n";
        ss << "Time Step: " << std::setw(3) << iter << "\n";
        ss << "----------------------------------------\n";
        ss << "Particle Move: " << std::fixed << std::setprecision(3)
            << std::setw(6) << std::chrono::duration<double, std::milli>(t_particlemove_end - t_particlemove_start).count() << " ms\n";
        ss << "Reassign:      " << std::setw(6)
            << std::chrono::duration<double, std::milli>(t_ressign_end - t_ressign_start).count() << " ms\n";
        ss << "Collision:     " << std::setw(6)
            << std::chrono::duration<double, std::milli>(t_collision_end - t_collision_start).count() << " ms\n";
        ss << "Total:         " << std::setw(6)
            << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms\n";
        ss << "========================================\n";
        std::cout << ss.str();

        if(iter % 100 == 0){
            d_cells->Sample(d_particles->d_vel_x, d_particles->d_vel_y, d_particles->d_vel_z, d_particles->N);
            TransferCellsFromDeviceToHost();
            m_output->Write2VTK("./res/result_" + std::to_string(iter));
        }
    }

    TransferCellsFromDeviceToHost();
    m_output->Write2VTK("./res/finalresult");
}

void Run::finalize()
{   
    m_particles.clear();
    m_cells.clear();
    std::cout << "Simulation Finished" << std::endl;
}
