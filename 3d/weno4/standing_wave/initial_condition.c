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
    PetscReal k = 1;
    
    Q0.comp[0] = PetscSinReal(PETSC_PI*k*x)*PetscSinReal(PETSC_PI*k*y)*PetscSinReal(PETSC_PI*k*z);
    Q0.comp[1] = 0.0;
    
    return Q0; 
}

//----------------------------------------------------------------------------
// Exact solution  
//----------------------------------------------------------------------------

Field ExactSolution(PetscReal x, PetscReal y, PetscReal z, PetscReal t){               
 
    Field Q0;
    PetscReal k = 1;
    PetscReal omega = PetscSqrtReal(3.0*PETSC_PI*PETSC_PI*k*k);
    
    Q0.comp[0] = PetscSinReal(PETSC_PI*k*x)*PetscSinReal(PETSC_PI*k*y)*PetscSinReal(PETSC_PI*k*z)*PetscCosReal(omega*t);
    Q0.comp[1] = 0.0;
    
    return Q0;
} 
