/*
 * initial_condition.c
 *      Author: sunder
 */ 
#include "hype.h" 

//----------------------------------------------------------------------------
// Initial condition function 
//----------------------------------------------------------------------------

Field InitialCondition(PetscReal x, PetscReal y, PetscReal z) {

    Field Q0;
    
    PetscReal kx = PETSC_PI;
    PetscReal ky = PETSC_PI;

    Q0.comp[0] = sin(kx*x)*sin(ky*y);
    Q0.comp[1] = 0.0;

    return Q0; 
}

//----------------------------------------------------------------------------
// Exact solution  
//----------------------------------------------------------------------------

Field ExactSolution(PetscReal x, PetscReal y, PetscReal z, PetscReal t){               
 
    Field Q0;

    PetscReal kx = PETSC_PI;
    PetscReal ky = PETSC_PI;

    //dispersion relation:
    PetscReal omega = wave_speed*sqrt(kx*kx + ky*ky);

    Q0.comp[0] = sin(kx*x)*sin(ky*y)*cos(omega*t);
    Q0.comp[1] = -omega*sin(kx*x)*sin(ky*y)*sin(omega*t);

    return Q0;
} 
