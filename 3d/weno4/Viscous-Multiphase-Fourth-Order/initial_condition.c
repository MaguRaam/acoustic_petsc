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
    
    
    /*
    PetscReal rho0 = 1.0;
    PetscReal vlx0 = 1.0;
    PetscReal vly0 = 1.0;
    PetscReal prs0 = 1.0;

    PetscReal kappa = 5.0; // Strength of the vortex

    PetscReal r2 = x*x + z*z;
    PetscReal exp_r2 = PetscExpReal(0.5*(1.0 - r2));

    PetscReal tempaa = -exp_r2*exp_r2*kappa*kappa*(GAMMA - 1.0)/(8.0*GAMMA*PETSC_PI*PETSC_PI);
    PetscReal tempab = tempaa + prs0/rho0;

    tempab = tempab*PetscPowReal(rho0,GAMMA)/prs0;

    V0.comp[0] = PetscPowReal(tempab, ( 1.0 / (GAMMA - 1.0)));
    V0.comp[1] = vlx0 - z*kappa *exp_r2/(2.0*PETSC_PI);
    V0.comp[3] = vly0 + x*kappa*exp_r2/(2.0*PETSC_PI);
    V0.comp[2] = 0.0; 
    V0.comp[4] = V0.comp[0]*(tempaa + prs0/rho0);
    */
    
    if (z < -0.25) {
        V0.comp[0] = 1.3765;
        V0.comp[1] = 0.0;
        V0.comp[2] = 0.0;
        V0.comp[3] = 0.3948; 
        V0.comp[4] = 1.57;
        V0.comp[5] = 1.0; 
    }
    
    else if (-0.25 <= z && z < -0.1) {
        V0.comp[0] = 1.0;
        V0.comp[1] = 0.0;
        V0.comp[2] = 0.0;
        V0.comp[3] = 0.0; 
        V0.comp[4] = 1.0;
        V0.comp[5] = 1.0; 
    }
    
    else if (-0.1 <= z && z < 0.1) {
        V0.comp[0] = 0.138;
        V0.comp[1] = 0.0;
        V0.comp[2] = 0.0;
        V0.comp[3] = 0.0; 
        V0.comp[4] = 1.0;
        V0.comp[5] = 0.0; 
    }
    
    else {
        V0.comp[0] = 1.0;
        V0.comp[1] = 0.0;
        V0.comp[2] = 0.0;
        V0.comp[3] = 0.0; 
        V0.comp[4] = 1.0;
        V0.comp[5] = 1.0; 
    }
    
    /*
    
    V0.comp[0] = 1.0 + 0.2*PetscSinReal(PETSC_PI*(x+y+z));
    V0.comp[1] = 1.0;
    V0.comp[2] = 1.0;
    V0.comp[3] = 1.0; 
    V0.comp[4] = 1.0;
    */
    //----------------------------------------------------------------------- 

    PDEPrim2Cons(&V0, &Q0);

    return Q0; 
}

/* Test Cases 

*/
