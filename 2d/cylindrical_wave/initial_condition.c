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
    PetscReal A = 1.0, sigma = 0.1;
    PetscReal r_2     =   x*x + y*y;
    PetscReal sigma_2 =   sigma*sigma;

    Q0[0] = A*PetscExpReal(-0.5*(r_2/sigma_2));
    Q0[1] = 0.0;
}

//----------------------------------------------------------------------------
// Exact solution 
//----------------------------------------------------------------------------

void ExactSolution(PetscReal x, PetscReal y, PetscReal t, PetscReal* Q0){
 
    PetscReal A = 1.0, sigma = 0.1;
    
    PetscReal r       = PetscSqrtReal(x*x + y*y);
    PetscReal r_2     =  (r - wave_speed*t)*(r - wave_speed*t);
    PetscReal sigma_2 =   sigma*sigma;

    Q0[0] = (A/r)*PetscExpReal(-0.5*(r_2/sigma_2)   );
    Q0[1] = 0.0;
}   