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
    
    //Radially symmetric Gaussian pulse:

    PetscReal A = 1.0;      //amplitude of the Gaussian
    PetscReal sigma = 0.1;  
    PetscReal r_2     =   x*x + y*y + z*z; //distance square
    PetscReal sigma_2 =   sigma*sigma;     

    Q0.comp[0] = A*PetscExpReal(-0.5*(r_2/sigma_2));
    Q0.comp[1] = 0.0;

    return Q0; 
}

