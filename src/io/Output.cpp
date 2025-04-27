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

