#pragma once

//hdf5
#include <H5Cpp.h>
//vtk
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkXMLPStructuredGridWriter.h>

#include <string>
#include <ostream>
#include "../Run.h"

class Run;
class Output
{
public:
    Output() = default;
    Output(Run* run);

    void Write2HDF5(const std::string& filename);
    void Write2VTK(const std::string& filename);
    void flattenCells(Cell* cell, std::vector<Cell*>& flatlist);
    void WriteAMRmesh(const std::string& filename);
protected:
    Run* m_run {nullptr};
};