/*
 * monitor_function.c
 *      Author: sunder
 */ 

#include "hype.h"  

//----------------------------------------------------------------------------
// Monitor function for additional processing in the intermediate time steps 
//----------------------------------------------------------------------------

PetscErrorCode MonitorFunction (TS ts,PetscInt step, PetscReal time, Vec U, void *ctx) {

    PetscErrorCode ierr; 
    AppCtx *Ctx = (AppCtx*)ctx;

    ierr = PetscPrintf(PETSC_COMM_WORLD,"%d t = %f\n", step, time); CHKERRQ(ierr);

    // TODO Plot the solution at the required time interval 

    /*if (Ctx->WriteInterval != 0) {

        if(step%Ctx->WriteInterval == 0) {
        
            DM da;
            ierr = TSGetDM(ts,&da); CHKERRQ(ierr);
            
            char filename[20];
            sprintf(filename, "plot/sol-%05d.vtk", step); // 4 is the padding level, increase it for longer simulations 
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data in vtk format to %s at t = %f, step = %d\n", filename, time, step); CHKERRQ(ierr);
            PetscViewer viewer;  
            PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);
            PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);
            ierr = DMView(da, viewer);
            VecView(U, viewer);
            
            ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
        
        }
    }*/

    return ierr; 
} 
