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
    PetscReal k = 1;
    Q0[0] = PetscSinReal(PETSC_PI*k*x)*PetscSinReal(PETSC_PI*k*y);
    Q0[1] = 0.0;
}

//----------------------------------------------------------------------------
// Exact solution 
//----------------------------------------------------------------------------

void ExactSolution(PetscReal x, PetscReal y, PetscReal t, PetscReal* Q0){
 
    PetscReal k = 1;
    PetscReal omega = PetscSqrtReal(2.0*PETSC_PI*PETSC_PI*k*k);

    Q0[0] = PetscSinReal(PETSC_PI*k*x)*PetscSinReal(PETSC_PI*k*y)*PetscCosReal(omega*t);
    Q0[1] = 0.0;
}   