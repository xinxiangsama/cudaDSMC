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
    const hsize_t N1 = static_cast<hsize_t>(m_run->m_mesh->getnumberCellsX());
    const hsize_t N2 = static_cast<hsize_t>(m_run->m_mesh->getnumberCellsY());

    // 打开文件（串行）
    H5::H5File file((filename + ".h5").c_str(), H5F_ACC_TRUNC);

    hsize_t dims[2] = {N1, N2};
    H5::DataSpace dataspace(2, dims);

    // 创建数据集属性
    H5::DSetCreatPropList plist;
    plist.setChunk(2, dims);

    // 创建数据集
    H5::DataSet dataset_U = file.createDataSet("U", H5::PredType::NATIVE_DOUBLE, dataspace, plist);
    H5::DataSet dataset_V = file.createDataSet("V", H5::PredType::NATIVE_DOUBLE, dataspace, plist);
    H5::DataSet dataset_W = file.createDataSet("W", H5::PredType::NATIVE_DOUBLE, dataspace, plist);
    H5::DataSet dataset_P = file.createDataSet("P", H5::PredType::NATIVE_DOUBLE, dataspace, plist);
    H5::DataSet dataset_T = file.createDataSet("T", H5::PredType::NATIVE_DOUBLE, dataspace, plist);
    H5::DataSet dataset_Rho = file.createDataSet("Rho", H5::PredType::NATIVE_DOUBLE, dataspace, plist);
    H5::DataSet dataset_numchild = file.createDataSet("Numchild", H5::PredType::NATIVE_INT, dataspace, plist);
    H5::DataSet dataset_numgrandchild = file.createDataSet("Numgrandchild", H5::PredType::NATIVE_INT, dataspace, plist);
    H5::DataSet dataset_mfp = file.createDataSet("MFP", H5::PredType::NATIVE_DOUBLE, dataspace, plist);

    // 准备数据
    std::vector<double> U(N1 * N2, 0.0);
    std::vector<double> V(N1 * N2, 0.0);
    std::vector<double> W(N1 * N2, 0.0);
    std::vector<double> P(N1 * N2, 0.0);
    std::vector<double> T(N1 * N2, 0.0);
    std::vector<double> Rho(N1 * N2, 0.0);
    std::vector<int> Numchild(N1 * N2, 0);
    std::vector<int> Numgrandchild(N1 * N2, 0);
    std::vector<double> MFP(N1 * N2, 0.0);

    for (size_t i = 0; i < N1; ++i){
        for (size_t j = 0; j < N2; ++j){
            size_t idx = i * N2 + j;
            auto cell = m_run->m_cells[idx];
            auto phase = cell.getphase();

            U[idx] = phase->getvelocity()[0];
            V[idx] = phase->getvelocity()[1];
            W[idx] = phase->getvelocity()[2];
            P[idx] = phase->getpressure();
            T[idx] = phase->gettemperature();
            Rho[idx] = phase->getdensity();
            Numchild[idx] = cell.getchildren().size();
            for (auto &child : cell.getchildren()) {
                Numgrandchild[idx] += child->getchildren().size();
            }
            MFP[idx] = cell.getmfp();
        }
    }

    // 写入数据
    dataset_U.write(U.data(), H5::PredType::NATIVE_DOUBLE);
    dataset_V.write(V.data(), H5::PredType::NATIVE_DOUBLE);
    dataset_W.write(W.data(), H5::PredType::NATIVE_DOUBLE);
    dataset_P.write(P.data(), H5::PredType::NATIVE_DOUBLE);
    dataset_T.write(T.data(), H5::PredType::NATIVE_DOUBLE);
    dataset_Rho.write(Rho.data(), H5::PredType::NATIVE_DOUBLE);
    dataset_numchild.write(Numchild.data(), H5::PredType::NATIVE_INT);
    dataset_numgrandchild.write(Numgrandchild.data(), H5::PredType::NATIVE_INT);
    dataset_mfp.write(MFP.data(), H5::PredType::NATIVE_DOUBLE);
}


void Output::Write2VTK(const std::string &filename)
{
    auto N1 = m_run->m_mesh->getnumberCellsX();
    auto N2 = m_run->m_mesh->getnumberCellsY();
    auto dx = m_run->m_mesh->getUnidX();
    auto dy = m_run->m_mesh->getUnidY();

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
