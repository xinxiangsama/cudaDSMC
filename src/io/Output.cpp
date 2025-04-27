#include "Output.h"
#include <vtkUnstructuredGrid.h>
#include <vtkQuad.h>
#include <vtkCellData.h>
#include <vtkXMLUnstructuredGridWriter.h>

Output::Output(Run *run)
{
    m_run = run;
}

void Output::Write2HDF5(const std::string &filename)
{

}

void Output::Write2VTK(const std::string &filename)
{
    auto N1 = m_run->m_mesh->getnumberCellsX();
    auto N2 = m_run->m_mesh->getnumberCellsY();
    auto dx = m_run->m_mesh->getUnidX();
    auto dy = m_run->m_mesh->getUnidX();

    auto& local_elements = m_run->m_mesh->getElements();
    // create a mesh obj
    vtkSmartPointer<vtkStructuredGrid> local_grid = vtkSmartPointer<vtkStructuredGrid>::New();
    local_grid->SetDimensions(N1, N2, 1);

    // create a points set
    vtkSmartPointer<vtkPoints> local_points = vtkSmartPointer<vtkPoints>::New();
    for(auto& element : local_elements){
        auto& vertices = element->getvertices();
        
        auto x = vertices[0]->getPosition()[0];
        auto y = vertices[0]->getPosition()[1];
        double z = 0.0;
        local_points->InsertNextPoint(x, y, z);
    }

    local_grid->SetPoints(local_points);

    std::vector<double> U(N1 * N2, 0.0);
    std::vector<double> V(N1 * N2, 0.0);
    std::vector<double> W(N1 * N2, 0.0);
    std::vector<double> P(N1 * N2, 0.0);
    std::vector<double> T(N1 * N2, 0.0);
    std::vector<double> Rho(N1 * N2, 0.0);
    for (size_t i = 0; i < N1; ++i){
        for (size_t j = 0; j < N2; ++j){
            auto cell = m_run->m_cells[i * N2 + j];
            auto phase = cell.getphase();
            auto element = cell.getelement();

            U[i * N2 + j] = phase->getvelocity()[0];
            V[i * N2 + j] = phase->getvelocity()[1];
            W[i * N2 + j] = phase->getvelocity()[2];
            P[i * N2 + j] = phase->getpressure();
            T[i * N2 + j] = phase->gettemperature();
            Rho[i * N2 + j] = phase->getdensity();
        }
    }

    // 创建并添加标量场数据
    auto addScalarField = [&](const std::vector<double>& data, const std::string& name) {
        vtkSmartPointer<vtkDoubleArray> array = vtkSmartPointer<vtkDoubleArray>::New();
        array->SetName(name.c_str());
        array->SetNumberOfComponents(1);
        array->SetNumberOfTuples(N1 * N2);
        for (size_t idx = 0; idx < data.size(); ++idx) {
            array->SetValue(idx, data[idx]);
        }
        local_grid->GetPointData()->AddArray(array);
    };

    
    addScalarField(U, "U");
    addScalarField(V, "V");
    addScalarField(P, "P");
    addScalarField(T, "T");
    addScalarField(Rho, "Rho");

    // wtite in vtk
    vtkSmartPointer<vtkXMLStructuredGridWriter> writer = vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
    writer->SetFileName((filename + ".vts").c_str());
    // writer->SetDataModeToAscii(); 
    writer->SetDataModeToBinary();  
    writer->SetInputData(local_grid);
    writer->Write();
}
