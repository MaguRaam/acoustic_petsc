/*
 * initial_condition.c
 *      Author: sunder
 */ 
#include "hype.h" 


//----------------------------------------------------------------------------
// Initial condition function 
//----------------------------------------------------------------------------

void InitialCondition(PetscReal x, PetscReal y, PetscReal *Q0)
{
    PetscReal kx = 2*PETSC_PI;
    PetscReal ky = 2*PETSC_PI;

    Q0[0] = sin(kx*x)*sin(ky*y);
    Q0[1] = 0.0;
}

//----------------------------------------------------------------------------
// Exact solution 
//----------------------------------------------------------------------------

void ExactSolution(PetscReal x, PetscReal y, PetscReal t, PetscReal* Q0){
 
    PetscReal kx = 2*PETSC_PI;
    PetscReal ky = 2*PETSC_PI;

    //dispersion relation:
    PetscReal omega = wave_speed*sqrt(kx*kx + ky*ky);

    Q0[0] = sin(kx*x)*sin(ky*y)*cos(omega*t);
    Q0[1] = -omega*sin(kx*x)*sin(ky*y)*sin(omega*t);
}   
