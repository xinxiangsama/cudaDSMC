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
    bool ifdisfuse = false;
    // inlet = std::make_unique<PeriodicBoundary>(Eigen::Vector3d(0.0, 0.5 * L2, 0.5 * L3), Eigen::Vector3d(1.0, 0.0, 0.0), 0, L1);
    inlet = std::make_unique<InletBoundary>(1);
    // outlet = std::make_unique<PeriodicBoundary>(Eigen::Vector3d(L1, 0.5 * L2, 0.5 * L3), Eigen::Vector3d(-1.0, 0.0, 0.0), 0, L1);
    outlet = std::make_unique<OutletBoundary>(Eigen::Vector3d(L1, 0.5 * L2, 0.5 * L3), Eigen::Vector3d(-1.0, 0.0, 0.0));
    // inlet = std::make_unique<WallBoundary>(Eigen::Vector3d(0.0, 0.5 * L2, 0.5 * L3), Eigen::Vector3d(1.0, 0.0, 0.0), ifdisfuse);
    // outlet = std::make_unique<WallBoundary>(Eigen::Vector3d(L1, 0.5 * L2, 0.5 * L3), Eigen::Vector3d(-1.0, 0.0, 0.0), ifdisfuse);
    // wall1 = std::make_unique<WallBoundary>(Eigen::Vector3d(0.5 * L1, 0.0, 0.5 * L3), Eigen::Vector3d(0.0, 1.0, 0.0), ifdisfuse);
    // wall2 = std::make_unique<WallBoundary>(Eigen::Vector3d(0.5 * L1, L2, 0.5 * L3), Eigen::Vector3d(0.0, -1.0, 0.0), ifdisfuse);
    wall1 = std::make_unique<OutletBoundary>(Eigen::Vector3d(0.5 * L1, 0.0, 0.5 * L3), Eigen::Vector3d(0.0, 1.0, 0.0));
    wall2 = std::make_unique<OutletBoundary>(Eigen::Vector3d(0.5 * L1, L2, 0.5 * L3), Eigen::Vector3d(0.0, -1.0, 0.0));
    // wall3 = std::make_unique<WallBoundary>(Eigen::Vector3d(0.5 * L1, 0.5 * L2, 0.0), Eigen::Vector3d(0.0, 0.0, 1.0), ifdisfuse);
    // wall4 = std::make_unique<WallBoundary>(Eigen::Vector3d(0.5 * L1, 0.5 * L2, L3), Eigen::Vector3d(0.0, 0.0, -1.0), ifdisfuse);
    wall3 = std::make_unique<PeriodicBoundary>(Eigen::Vector3d(0.5 * L1, 0.5 * L2, 0.0), Eigen::Vector3d(0.0, 0.0, 1.0), 2, L3);
    wall4 = std::make_unique<PeriodicBoundary>(Eigen::Vector3d(0.5 * L1, 0.5 * L2, L3), Eigen::Vector3d(0.0, 0.0, -1.0), 2, L3);
    
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
    for(int i = 0; i < 100; ++i){
        d_particles->Move(tau, 128);
    }
    // for(auto& cell : m_cells){
    //     cell.sample();
    // }
    // m_output->Write2HDF5("./res/init.h5");

    d_cells->CalparticleNum(d_particles->d_pos_x, d_particles->d_pos_y, d_particles->d_pos_z, d_particles->local_id, d_particles->cell_id,d_particles->N, N1, N2, N3, 128);
    d_cells->CalparticleStartIndex();

    d_particles->Sort();
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

    for(auto& cell : m_cells){
        auto particles = cell.getparticles();
        for(auto& particle : particles){
            // cell.comtimetokenleaving(particle); 
            // auto dt = cell.getdt();
            particle->Move(tau);

            if(cell.ifcut()){
                for(auto& segment : cell.getelement()->getsegments()){
                    if(segment->isHit(particle->getposition())){
                        segment->Reflect(particle, tau);
                    }
                }
            }

            if(inlet->isHit(particle->getposition())){
                inlet->Reflect(particle, tau);
            }
            if(outlet->isHit(particle->getposition())){
                outlet->Reflect(particle, tau);
            }

            if(wall1->isHit(particle->getposition())){
                wall1->Reflect(particle, tau);
            }
            if(wall2->isHit(particle->getposition())){
                wall2->Reflect(particle, tau);
            }

            if(wall3->isHit(particle->getposition())){
                wall3->Reflect(particle, tau);
            }
            if(wall4->isHit(particle->getposition())){
                wall4->Reflect(particle, tau);
            }

        }
    }
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

void Run::ressignParticle()
{   
    for(auto& cell : m_cells){
        cell.removeallparticles();
    }

    inlet->InjetParticle(m_particles);
    auto t_classify_start = std::chrono::high_resolution_clock::now();
    std::vector<Particle> particle_out;
    particle_out.reserve(m_particles.size());
    size_t i = 0;
    while (i < m_particles.size()) {
        auto& particle = m_particles[i];
        if (!particle.ifvalid()) {
            // 无效粒子，直接用最后一个覆盖并 pop_back 掉队尾被移动掉的粒子
            m_particles[i] = std::move(m_particles.back());
            m_particles.pop_back();
            continue;
        }
    
        ++i; // 只在当前粒子保留时才前进
    }
    auto t_classify_end = std::chrono::high_resolution_clock::now();



    auto t_assign_start = std::chrono::high_resolution_clock::now();
    assignParticle2cell();
    auto t_assign_end = std::chrono::high_resolution_clock::now();

    int N_particle_local = m_particles.size();
    int N_particle_global {};

    for(auto& cell : m_cells){
        cell.sortParticle2children();
        // std::cout << "sort particle to children done!"<<std::endl;
    }
}

void Run::collision()
{
    int Ncollision_local {};
    for(auto& cell : m_cells){
        cell.collision();
        Ncollision_local += cell.getCollisionNum();
    }

    int Ncollision_global {};

}

void Run::assignParticle2cell()
{
    for(auto& particle : m_particles){
        auto cellID = m_mesh->getIndex(particle.getposition());
        particle.setcellID(cellID);
        m_cells[cellID].insertparticle(&particle);
    }
}

void Run::solver()
{
    for(size_t iter = 0; iter < 5000; ++iter){

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
        if (iter % 100 == 0) {
            for(auto& cell : m_cells){
                cell.sample();
                // cell.VTS();
                // cell.genAMRmesh();
                // std::cout << "gen amr done!"<<std::endl;
                // cell.sortParticle2children();
                // std::cout << "sort particle to children done!"<<std::endl; //shouldnt be here!
            }
            m_output->Write2HDF5("./res/step" + std::to_string(iter) + ".h5");
            // m_output->WriteAMRmesh("./res/step" + std::to_string(iter) +"AMRmesh"+".h5");
            // m_output->Write2VTK("./res/step" + std::to_string(iter));
            // m_output->WriteAMR2VTK("./res/step" + std::to_string(iter) +"AMRmesh");
        }

        // if(iter == 0 || iter == 500 || iter == 1000 || iter == 2000 || iter == 4000){
        //     for(auto& cell : m_cells){
        //         cell.sample();
        //         cell.VTS();
        //         cell.genAMRmesh();
        //     }
        // }
    }
    std::cout << "Simulation Finished" << std::endl;
    m_output->Write2HDF5("./res/final.h5");
    std::cout << "Output Finished" << std::endl;
}

void Run::finalize()
{
    m_cells.clear();
}
