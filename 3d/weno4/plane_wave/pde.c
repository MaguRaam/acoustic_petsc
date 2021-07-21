/*
 * pde.c
 *      Author: sunder
 */

#include "hype.h"

//----------------------------------------------------------------------------
// Viscous part of the flux in the normal direction 
//----------------------------------------------------------------------------

void PDEFlux(const PetscReal grad_Q[nVar][DIM], PetscReal nx, PetscReal ny, PetscReal nz, PetscReal* F) {
    
    *F = wave_speed*wave_speed*(grad_Q[0][0]*nx + grad_Q[0][1]*ny + grad_Q[0][2]*nz); 

}

//----------------------------------------------------------------------------
// Viscous Riemann Solver (Does average of the two fluxes)  
//----------------------------------------------------------------------------

void RiemannSolver(PetscReal grad_QL[nVar][DIM], 
                      PetscReal grad_QR[nVar][DIM], 
                      const PetscReal nx, const PetscReal ny, const PetscReal nz,
                      PetscReal* Flux) {
    
    PetscReal FL, FR; 
    
    PDEFlux(grad_QL, nx, ny, nz, &FL); 
    PDEFlux(grad_QR, nx, ny, nz, &FR);
    
    *Flux = 0.5*(FR + FL);

}
