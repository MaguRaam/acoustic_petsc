/*
 * initialize_solution.c
 *      Author: sunder
 */ 

#include "hype.h" 

//----------------------------------------------------------------------------
// Initialize the solution with the given initial condition  
//----------------------------------------------------------------------------

PetscErrorCode InitializeSolution(Vec U, DM da, AppCtx Ctx) {

    PetscErrorCode ierr;

    DM          coordDA;
    Vec         coordinates;
    DMDACoor3d  ***coords;
    Field  ***u;
    PetscInt    xs, ys, zs, xm, ym, zm, i, j, k, c, l, m, n;
    PetscReal integral[nVar]; 
    PetscReal xc, yc, zc, xGP, yGP, zGP;
    PetscReal h = Ctx.h; 

    Field Q0; 

    ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(da, &coordDA);CHKERRQ(ierr);
    ierr = DMGetCoordinates(da, &coordinates);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, U, &u);CHKERRQ(ierr);

    // Use five point gauss quadrature

    for (k = zs; k < zs+zm; ++k) {
        for (j = ys; j < ys+ym; ++j) {
            for (i = xs; i < xs+xm; ++i) {
            
                // Get coordinates of corner of the cell
            
                xc = coords[k][j][i].x - 0.5*h; 
                yc = coords[k][j][i].y - 0.5*h;
                zc = coords[k][j][i].z - 0.5*h;
            
                for (c = 0; c < nVar; ++c)
                    integral[c] = 0.0;
            
                for(l = 0; l < N_gp5; l++) {
                    for (m = 0; m < N_gp5; m++) {
                        for (n = 0; n < N_gp5; ++n) {
                            
                            xGP = xc + (PetscReal)l*h*x_gp5[l];
                            yGP = yc + (PetscReal)m*h*x_gp5[m];
                            zGP = zc + (PetscReal)n*h*x_gp5[n];
                            
                            Q0 = InitialCondition(xGP,yGP,zGP);
                            
                            for (c = 0; c < nVar; ++c) {
                                integral[c] += w_gp5[l]*w_gp5[m]*w_gp5[n]*Q0.comp[c];
                            }
                        }
                    }
                }
            
                for (c = 0; c < nVar; ++c) {
                    u[k][j][i].comp[c] = integral[c];
                }
            }
        }
    }

    ierr = DMDAVecRestoreArray(da, U, &u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);

    return ierr; 
}
 
