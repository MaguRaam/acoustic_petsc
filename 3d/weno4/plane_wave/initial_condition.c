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
    
    PetscReal kx = 1.0;
    PetscReal ky = 1.0;
    PetscReal kz = 1.0;

    //dispersion relation:
    PetscReal omega = wave_speed*sqrt(kx*kx + ky*ky + kz*kz);

    Q0.comp[0] = sin(kx*x + ky*y + kz*z);
    Q0.comp[1] = -omega*cos(kx*x + ky*y + kz*z);

    return Q0; 
}

//----------------------------------------------------------------------------
// Exact solution  
//----------------------------------------------------------------------------

Field ExactSolution(PetscReal x, PetscReal y, PetscReal z, PetscReal t){               
 
    Field Q0;

    PetscReal kx = 1.0;
    PetscReal ky = 1.0;
    PetscReal kz = 1.0;

    //dispersion relation:
    PetscReal omega = wave_speed*sqrt(kx*kx + ky*ky + kz*kz);

    Q0.comp[0] = sin(kx*x + ky*y + kz*z - omega*t);
    Q0.comp[1] = -omega*cos(kx*x + ky*y + kz*z - omega*t);

    return Q0;
} 
