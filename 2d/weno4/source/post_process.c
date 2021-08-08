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
    
    return ierr; 
}
