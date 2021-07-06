/*
 * post_process.c
 *      Author: sunder
 */ 

#include "hype.h"  

//----------------------------------------------------------------------------
// Monitor function for additional processing in the intermediate time steps 
//----------------------------------------------------------------------------

PetscErrorCode MonitorFunction (TS ts,PetscInt step, PetscReal time, Vec U, void *ctx) {

    PetscErrorCode ierr; 
    AppCtx *Ctx = (AppCtx*)ctx;

    // Set the time step based on CFL condition 

    ierr = PetscPrintf(PETSC_COMM_WORLD,"%d t = %.5e\n", step, time);CHKERRQ(ierr);

    // Plot the solution at the required time interval 

    if (Ctx->WriteInterval != 0) {

        if(step%Ctx->WriteInterval == 0) {
            
            DM da;
            ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
            
            char filename[20];
            sprintf(filename, "plot/sol-%08d.vts", step); // 8 is the padding level, increase it for longer simulations 
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data in vts format to %s at t = %f, step = %d\n", filename, time, step);CHKERRQ(ierr);
            PetscViewer viewer;  
            ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr); 
            ierr = DMView(da, viewer);
            VecView(U, viewer);
            ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
            
        }
    }

    if (Ctx->RestartInterval != 0) {
        
        PetscViewer viewer_binary;

        if(step%Ctx->RestartInterval == 0) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data in binary to restart1.bin at t = %f\n", time);CHKERRQ(ierr);
            ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"restart1.bin",FILE_MODE_WRITE, &viewer_binary);CHKERRQ(ierr);
            ierr = VecView(U,viewer_binary);CHKERRQ(ierr);
            ierr = PetscViewerDestroy(&viewer_binary); CHKERRQ(ierr);
        }
        
        if((step+10)%(Ctx->RestartInterval) == 0) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data in binary to restart2.bin at t = %f\n", time);CHKERRQ(ierr);
            ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"restart2.bin",FILE_MODE_WRITE, &viewer_binary);CHKERRQ(ierr);
            ierr = VecView(U,viewer_binary);CHKERRQ(ierr);
            ierr = PetscViewerDestroy(&viewer_binary);CHKERRQ(ierr);
        }
    }
    
    return ierr; 
}

//----------------------------------------------------------------------------
// Find L2 and L_inf errors for periodic test cases with square domain and 
// where final solution conicides with the initial condition
//----------------------------------------------------------------------------

PetscErrorCode ErrorNorms(Vec U, DM da, AppCtx Ctx, PetscReal* l2, PetscReal* linf, PetscReal t) {

    PetscErrorCode ierr;

    DM          coordDA;
    Vec         coordinates;
    DMDACoor2d  **coords;
    Field   **u;
    Field   **u_exact;
    PetscInt    xs, ys, xm, ym, i, j, c, l , m;
    PetscReal integral[nVar]; 
    PetscReal xc, yc, xGP, yGP;
    PetscReal h = Ctx.h;
    PetscReal Q0[nVar];
    Vec U_exact;
    
    ierr = VecDuplicate(U,&U_exact);CHKERRQ(ierr);
    ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(da, &coordDA);CHKERRQ(ierr);
    ierr = DMGetCoordinates(da, &coordinates);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, U, &u);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, U_exact, &u_exact);CHKERRQ(ierr);

    // Use five point gauss quadrature

    for (j = ys; j < ys+ym; ++j) {
        for (i = xs; i < xs+xm; ++i) {
            
            // Get coordinates of center of the cell 
            
            xc = coords[j][i].x; 
            yc = coords[j][i].y;
            
            for (c = 0; c < nVar; ++c)
                integral[c] = 0.0;
            
            for(l = 0; l < N_gp5; ++l) {
                for (m = 0; m < N_gp5; ++m) {

                    xGP = xc + h*x_gp5[l];
                    yGP = yc + h*x_gp5[m];
                    
                    ExactSolution(xGP,yGP,t,Q0);
                    
                    for (c = 0; c < nVar; ++c) 
                        integral[c] += w_gp5[l]*w_gp5[m]*Q0[c];
                }
            }
            
            for (c = 0; c < nVar; ++c) {
                if (c == 0) {
                    u_exact[j][i].comp[c] = integral[c]; 
                }
                
                else {
                    u[j][i].comp[c] = 0.0;
                    u_exact[j][i].comp[c] = 0.0; 
                }
            }
        }
    }

    ierr = DMDAVecRestoreArray(da, U, &u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, U_exact, &u_exact);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);

    PetscReal nrm_inf, nrm2; 

    ierr = VecAXPY(U_exact, -1.0, U);CHKERRQ(ierr);
    ierr = VecNorm(U_exact, NORM_INFINITY, &nrm_inf);CHKERRQ(ierr);
    ierr = VecNorm(U_exact, NORM_2, &nrm2);CHKERRQ(ierr);

    nrm2 = nrm2/((PetscReal)Ctx.N_x);

    *l2 = nrm2; 
    *linf = nrm_inf; 

    ierr = VecDestroy(&U_exact);CHKERRQ(ierr);

    return ierr; 
}


