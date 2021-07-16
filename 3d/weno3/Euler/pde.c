#include "hype.h"

/* Definitions of various PDE functions */

//----------------------------------------------------------------------------
// Convert a conserved variable to primitive variable
//----------------------------------------------------------------------------

void PDECons2Prim(const Field* Q, Field* V) {
    
    PetscReal r1_rho = 1.0/Q->comp[0]; 
    
    V->comp[0] = Q->comp[0];
    V->comp[1] = r1_rho*Q->comp[1];
    V->comp[2] = r1_rho*Q->comp[2];
    V->comp[3] = r1_rho*Q->comp[3];
    V->comp[4] = (GAMMA -1.0)*( Q->comp[4] - 0.5*r1_rho*( (Q->comp[1]*Q->comp[1] + Q->comp[2]*Q->comp[2] + Q->comp[3]*Q->comp[3]) ) );
}

//----------------------------------------------------------------------------
// Convert a primitive variable to conserved variable
//----------------------------------------------------------------------------

void PDEPrim2Cons(const Field* V, Field* Q) {
    
    PetscReal e = V->comp[4]/(GAMMA - 1.0);
    PetscReal k = 0.5*V->comp[0]*(V->comp[1]*V->comp[1] + V->comp[2]*V->comp[2] + V->comp[3]*V->comp[3]);

    Q->comp[0] = V->comp[0];
    Q->comp[1] = V->comp[0]*V->comp[1];
    Q->comp[2] = V->comp[0]*V->comp[2];
    Q->comp[3] = V->comp[0]*V->comp[3]; 
    Q->comp[4] = k + e;
}

//----------------------------------------------------------------------------
// Find the conservative flux components F in the given normal direction
//----------------------------------------------------------------------------

PetscReal PDEConsFlux (const Field* Q, 
                       const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                       const PetscReal x,  const PetscReal y,  const PetscReal z, 
                       Field* F) {


    PetscReal rho = Q->comp[0];
    PetscReal r1_rho = 1.0/rho; 
    PetscReal u = r1_rho*Q->comp[1];
    PetscReal v = r1_rho*Q->comp[2];
    PetscReal w = r1_rho*Q->comp[3];
    PetscReal E = Q->comp[4]; 
    PetscReal p = (GAMMA -1.0)*( E - 0.5*rho*(u*u + v*v + w*w) );

    // Check if the input state is physically admissible 

    if (rho < rho_floor) {
	    printf("Negative density = %f\n", rho);
	    printf("At x = %f, y = %f, z = %f\n", x, y, z);  
	    MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    if (p < prs_floor) {
	    printf("Negative pressure = %f\n", p);
	    printf("At x = %f, y = %f, z = %f\n", x, y, z);  
	  	MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    // Now find the fluxes 

    F->comp[0] = nx*rho*u         + ny*rho*v         + nz*rho*w;
    F->comp[1] = nx*(rho*u*u + p) + ny*rho*u*v       + nz*rho*u*w;
    F->comp[2] = nx*rho*u*v       + ny*(rho*v*v + p) + nz*rho*v*w;
    F->comp[3] = nx*rho*u*w       + ny*rho*v*w       + nz*(rho*w*w + p);
    F->comp[4] = (E + p)*(nx*u + ny*v + nz*w);

    PetscReal a = PetscSqrtReal(r1_rho*GAMMA*p); 

    // Also obtain the maximum eigen value 

    PetscReal s_max = PetscAbsReal(u*nx + v*ny + w*nz) + a;

    return s_max;
}

//----------------------------------------------------------------------------
// Rusanov/LLF Riemann Solver 
//----------------------------------------------------------------------------

PetscReal PDELLFRiemannSolver(const Field* QL, const Field* QR, 
                              const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                              const PetscReal x, const PetscReal y,  const PetscReal z, 
                              Field* Flux) {
        
    Field FL, FR; PetscInt c; 

    PetscReal s_max_l = PDEConsFlux(QL, nx, ny, nz, x, y, z, &FL); 
    PetscReal s_max_r = PDEConsFlux(QR, nx, ny, nz, x, y, z, &FR);

    PetscReal s_max = PetscMax(s_max_l, s_max_r);

    for (c = 0; c < nVar; ++c) {
        Flux->comp[c] = 0.5*(FR.comp[c] + FL.comp[c] - s_max*(QR->comp[c] - QL->comp[c]));
    }

    return s_max; 
}
