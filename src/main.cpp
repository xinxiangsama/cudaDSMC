#include "./Run.h"
// #include <vtkVersion.h>

int main(int argc, char** argv)
{
    Run run;
    run.initialize(argc, argv);
    run.solver();
    run.finalize();
}