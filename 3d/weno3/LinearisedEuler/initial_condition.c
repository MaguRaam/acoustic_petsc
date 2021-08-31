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
    
    V0.comp[0] = 1.0;
    V0.comp[1] = 1.0;
    V0.comp[2] = 1.0;
    V0.comp[3] = 1.0; 

    //----------------------------------------------------------------------- 

    PDEPrim2Cons(&V0, &Q0);

    return Q0; 
}

/* Test Cases 

*/
