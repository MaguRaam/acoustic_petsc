/*
 * pde.c
 *      Author: sunder
 */

#include "hype.h"

//----------------------------------------------------------------------------
// Viscous part of the flux in the normal direction 
//----------------------------------------------------------------------------

void PDEFlux(const PetscReal grad_Q[nVar][DIM], PetscReal nx, PetscReal ny, PetscReal* F) {
    
    *F = wave_speed*wave_speed*(grad_Q[0][0]*nx + grad_Q[0][1]*ny); 

}

//----------------------------------------------------------------------------
// Viscous Riemann Solver (Does average of the two fluxes)  
//----------------------------------------------------------------------------

void LLFRiemannSolver(PetscReal grad_QL[nVar][DIM], 
                      PetscReal grad_QR[nVar][DIM], 
                      const PetscReal nx, const PetscReal ny,
                      PetscReal* Flux) {
    
    PetscReal FL, FR; 
    
    PDEFlux(grad_QL, nx, ny, &FL); 
    PDEFlux(grad_QR, nx, ny, &FR);
    
    *Flux = 0.5*(FR + FL);

}
