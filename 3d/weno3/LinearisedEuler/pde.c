#include "hype.h"

/* Definitions of various PDE functions */

//----------------------------------------------------------------------------
// Convert a conserved variable to primitive variable
//----------------------------------------------------------------------------

void PDECons2Prim(const Field *Q, Field *V)
{

    V->comp[0] = Q->comp[0];      
    V->comp[1] = Q->comp[1];      
    V->comp[2] = Q->comp[2];      
    V->comp[3] = Q->comp[3];      
}

//----------------------------------------------------------------------------
// Convert a primitive variable to conserved variable
//----------------------------------------------------------------------------

void PDEPrim2Cons(const Field *V, Field *Q)
{

    Q->comp[0] = V->comp[0];
    Q->comp[1] = V->comp[1];
    Q->comp[2] = V->comp[2];
    Q->comp[3] = V->comp[3];
}

//----------------------------------------------------------------------------
// Find the conservative flux components F in the given normal direction
//----------------------------------------------------------------------------

PetscReal PDEConsFlux(const Field *Q,
                      const PetscReal nx, const PetscReal ny, const PetscReal nz,
                      const PetscReal x, const PetscReal y, const PetscReal z,
                      Field *F)
{
    PetscReal K = rho_0 * c_0 * c_0;     //bulk modulus
    PetscReal invrho_0 = 1.0 / rho_0;

    PetscReal p = Q->comp[0];
    PetscReal u = Q->comp[1];
    PetscReal v = Q->comp[2];
    PetscReal w = Q->comp[3];

    // Check if the input state is physically admissible

    if (p < prs_floor)
    {
        printf("Negative pressure = %f\n", p);
        printf("At x = %f, y = %f, z = %f\n", x, y, z);
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    // Now find the fluxes                 

    F->comp[0] = nx * K * u + ny * K * v + nz * K * w;
    F->comp[1] = invrho_0 * nx * p;
    F->comp[2] = invrho_0 * ny * p;
    F->comp[3] = invrho_0 * nz * p;

    // Also obtain the maximum eigen value

    PetscReal s_max = c_0;   //! is the eigenvalue correct

    return s_max;
}

//----------------------------------------------------------------------------
// Rusanov/LLF Riemann Solver
//----------------------------------------------------------------------------

PetscReal PDELLFRiemannSolver(const Field *QL, const Field *QR,
                              const PetscReal nx, const PetscReal ny, const PetscReal nz,
                              const PetscReal x, const PetscReal y, const PetscReal z,
                              Field *Flux)
{

    Field FL, FR;
    PetscInt c;

    PetscReal s_max_l = PDEConsFlux(QL, nx, ny, nz, x, y, z, &FL);
    PetscReal s_max_r = PDEConsFlux(QR, nx, ny, nz, x, y, z, &FR);

    PetscReal s_max = PetscMax(s_max_l, s_max_r);

    for (c = 0; c < nVar; ++c)
    {
        Flux->comp[c] = 0.5 * (FR.comp[c] + FL.comp[c] - s_max * (QR->comp[c] - QL->comp[c]));
    }

    return s_max;
}
