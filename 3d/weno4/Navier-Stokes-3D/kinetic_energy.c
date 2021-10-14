/*
 * kinetic_energy.c
 *      Author: sunder
 */ 
#include "hype.h" 

//----------------------------------------------------------------------------
// Find the total kinetic energy in the domain 
//----------------------------------------------------------------------------

PetscErrorCode KineticEnergy(Vec U, DM da, PetscReal h, PetscReal* ke) {

    PetscErrorCode ierr=0;
    
    Field  ***u;
    PetscInt i, j, k, xs, ys, zs, xm, ym, zm;
    PetscReal ke_local = 0.0; 
    PetscReal ke_global; 
    PetscReal vol = h*h*h;

    ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da, U, &u);CHKERRQ(ierr);

    for (k = zs; k < zs+zm; ++k) {
        for (j = ys; j < ys+ym; ++j) {
            for (i = xs; i < xs+xm; ++i) {
            

                ke_local += vol*0.5*(u[k][j][i].comp[1]*u[k][j][i].comp[1] + u[k][j][i].comp[2]*u[k][j][i].comp[2] + u[k][j][i].comp[3]*u[k][j][i].comp[3])/
                            u[k][j][i].comp[0];
                            
            }
        }
    }

    ierr = DMDAVecRestoreArrayRead(da, U, &u);CHKERRQ(ierr);

    
    ierr = MPI_Allreduce(&ke_local, &ke_global,1,MPIU_REAL,MPIU_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
    //ierr = MPI_Allreduce(&ke_local, &ke_global,1,MPIU_REAL,MPIU_SUM, PetscObjectComm((PetscObject)da));CHKERRQ(ierr);


    *ke = ke_global; 
    
    return ierr; 
}  
