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
    Field V0;

    //----------------------------------------------------------------------- 
    
    PetscReal rho_0 = 1.0; 
    PetscReal p_0 = 100.0/GAMMA; 
    
    V0.comp[0] = rho_0;
    V0.comp[1] = PetscSinReal(x)*PetscCosReal(y)*PetscCosReal(z);
    V0.comp[2] = -PetscCosReal(x)*PetscSinReal(y)*PetscCosReal(z);
    V0.comp[3] = 0.0; 
    V0.comp[4] = p_0 + (rho_0/16.0)*(PetscCosReal(2.0*z) + 2.0)*(PetscCosReal(2.0*x) + PetscCosReal(2.0*y) - 2.0);
    
    //----------------------------------------------------------------------- 

    PDEPrim2Cons(&V0, &Q0);

    return Q0; 
}
