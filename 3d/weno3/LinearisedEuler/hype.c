/*
 * hype.c
 *      Author: sunder
 */ 

static char help[] = "Third Order 3D code for solving Euler equations using PETSc.\n\n";

#include "hype.h" 

//----------------------------------------------------------------------------
// Main function of the code 
//----------------------------------------------------------------------------

int main(int argc,char **argv) {

    // --------------------------------------------
    // Initialize MPI 
    //---------------------------------------------
    
    PetscErrorCode ierr;                    // For catching PETSc errors  
    PetscLogDouble start_time, end_time;    // For logging the time values 
    
    ierr = PetscInitialize(&argc, &argv, (char*)0, help);CHKERRQ(ierr);
    
    ierr =  PetscTime(&start_time);CHKERRQ(ierr);  
    
    // --------------------------------------------
    // Set important user defined parameters  
    //---------------------------------------------

    AppCtx Ctx; 

    Ctx.x_min           = -1.0;                            
    Ctx.x_max           =  1.0;                            
    Ctx.y_min           = -1.0;                           
    Ctx.y_max           =  1.0;
    Ctx.z_min           = -1.0; 
    Ctx.z_max           =  1.0;
    Ctx.N_x             =   64;
    Ctx.N_y             =   64;
    Ctx.N_z             =   64;
    Ctx.CFL             =  0.9; 
    Ctx.InitialStep     =  0; 
    Ctx.InitialTime     =  0.0;                            
    Ctx.FinalTime       =  2.0;                            
    Ctx.WriteInterval   =  1000;      
    Ctx.RestartInterval =  1000;
    Ctx.left_boundary   =  periodic;                   
    Ctx.right_boundary  =  periodic;                   
    Ctx.bottom_boundary =  periodic;                     
    Ctx.top_boundary    =  periodic;
    Ctx.front_boundary  =  periodic;                     
    Ctx.back_boundary   =  periodic;
    Ctx.ReconsPrimitive =  PETSC_FALSE; 
    Ctx.Restart         =  PETSC_FALSE; 
    Ctx.h = (Ctx.x_max - Ctx.x_min)/(PetscReal)(Ctx.N_x);  
    
    // --------------------------------------------
    // Data members  
    //---------------------------------------------

    Vec U;                           // Solution Vector (Conserved variables)
    Vec RHS;                         // RHS vector to update the solution
    DM da;                           // Grid object
    PetscInt time_steps;             // No. of time steps 
    TS ts;                           // Time stepping object 
    PetscMPIInt MyPID;               // Rank of the current processor 
    PetscMPIInt numProcs;            // Size of the communicator
    
    // --------------------------------------------
    // Obtain the rank of the process and size of 
    // the communicator 
    //---------------------------------------------

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&numProcs);CHKERRQ(ierr); 
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&MyPID);CHKERRQ(ierr); 
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Code running with %d processes\n", numProcs);CHKERRQ(ierr); 
    
    // --------------------------------------------
    // Initialize the grid and set field names
    //---------------------------------------------

    DMBoundaryType x_boundary;
    DMBoundaryType y_boundary;
    DMBoundaryType z_boundary; 
    
    if (Ctx.left_boundary == periodic || Ctx.right_boundary == periodic)
        x_boundary = DM_BOUNDARY_PERIODIC;
    else
        x_boundary = DM_BOUNDARY_GHOSTED; 

    if (Ctx.bottom_boundary == periodic || Ctx.top_boundary == periodic)
        y_boundary = DM_BOUNDARY_PERIODIC;
    else
        y_boundary = DM_BOUNDARY_GHOSTED;
    
    if (Ctx.front_boundary == periodic || Ctx.back_boundary == periodic)
        z_boundary = DM_BOUNDARY_PERIODIC;
    else
        z_boundary = DM_BOUNDARY_GHOSTED;

    ierr = DMDACreate3d(PETSC_COMM_WORLD,x_boundary,y_boundary,z_boundary,DMDA_STENCIL_BOX,Ctx.N_x,Ctx.N_y,Ctx.N_z,
                        PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,                             
                        nVar,3,NULL,NULL,NULL,&da);CHKERRQ(ierr);  
                        
    ierr = DMSetUp(da);CHKERRQ(ierr);
    
    // Now create various global vectors 

    ierr = DMCreateGlobalVector(da, &U); CHKERRQ(ierr);
    ierr = VecDuplicate(U,&RHS);         CHKERRQ(ierr);

    // Set coordinates of cell centers 
    
    ierr = DMDASetUniformCoordinates(da,Ctx.x_min + 0.5*Ctx.h, Ctx.x_max + 0.5*Ctx.h,
                                        Ctx.y_min + 0.5*Ctx.h, Ctx.y_max + 0.5*Ctx.h,
                                        Ctx.z_min + 0.5*Ctx.h, Ctx.z_max + 0.5*Ctx.h);CHKERRQ(ierr);
    // Set names of the fields

    ierr = DMDASetFieldName(da,0,"p");CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,1,"u");CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,2,"v");CHKERRQ(ierr);    
    ierr = DMDASetFieldName(da,3,"w");CHKERRQ(ierr);
    
    // --------------------------------------------
    // Allocate memory for boundary values and 
    // upwind fluxes
    //---------------------------------------------
    
    PetscInt xs,ys,xm,ym,zs,zm;
    
    ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
    
    Ctx.u_bnd = allocate6d(zm+2,ym+2, xm+2,nVar,6,N_gp2d); // 6 => number of faces in a cube
    Ctx.F     = allocate4d(zm,ym,xm+1,nVar);
    Ctx.G     = allocate4d(zm,ym+1,xm,nVar);
    Ctx.H     = allocate4d(zm+1,ym,xm,nVar);
    
    ierr = DMCreateLocalVector(da,&Ctx.localU);CHKERRQ(ierr);
    
    // --------------------------------------------
    // Initialize the solution (either with initial
    // condition or restart file)
    //---------------------------------------------
    
    if (Ctx.Restart) {
        
        // Initialize by reading the restart file 
        
        PetscViewer    viewer_binary;
        ierr = PetscPrintf(PETSC_COMM_WORLD,"reading vector in binary from restart1.bin ...\n");CHKERRQ(ierr);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"restart1.bin",FILE_MODE_READ,&viewer_binary);CHKERRQ(ierr);
        ierr = VecLoad(U,viewer_binary);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer_binary);CHKERRQ(ierr);
    }
    
    else {
        
        // Initialize by initial condition 
        
        ierr = InitializeSolution(U, da, Ctx);CHKERRQ(ierr);
    }
    
    ierr = InitializeSolution(U, da, Ctx);CHKERRQ(ierr);
    
    // --------------------------------------------
    // Advance solution in time   
    //---------------------------------------------
    
    ierr = TSCreate(PETSC_COMM_SELF, &ts);CHKERRQ(ierr);              
    ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);           
    ierr = TSSetDM(ts,da);CHKERRQ(ierr);                              
    
    if (Ctx.ReconsPrimitive) {
        //ierr = RHSFunctionPrimitive(ts, Ctx.InitialTime, U, RHS, &Ctx);CHKERRQ(ierr);
        //ierr = TSSetRHSFunction(ts,NULL,RHSFunctionPrimitive, &Ctx);CHKERRQ(ierr);
    }
    
    else {
        ierr = RHSFunction(ts, Ctx.InitialTime, U, RHS, &Ctx);CHKERRQ(ierr);
        ierr = TSSetRHSFunction(ts,NULL,RHSFunction, &Ctx);CHKERRQ(ierr);
    }
    
    ierr = TSSetStepNumber(ts, Ctx.InitialStep);CHKERRQ(ierr);
    ierr = TSSetTime(ts, Ctx.InitialTime);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, Ctx.dt); CHKERRQ(ierr);CHKERRQ(ierr);                     
    //ierr = TSMonitorSet(ts,MonitorFunction,&Ctx,NULL);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts, Ctx.FinalTime);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr); 
    ierr = TSSetType(ts, TSSSP);CHKERRQ(ierr); 
    ierr = TSSSPSetType(ts, TSSSPRKS3);CHKERRQ(ierr); 
    ierr = TSSSPSetNumStages(ts,4);CHKERRQ(ierr); 
    ierr = TSSolve(ts, U);CHKERRQ(ierr);
    ierr = TSGetStepNumber(ts,&time_steps);CHKERRQ(ierr); 
    
    // --------------------------------------------
    // Output solution in vtk format   
    //--------------------------------------------
    
    char filename[20]; 
    sprintf(filename, "sol-%05d.vtk", time_steps);
    PetscViewer viewer;  
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(da, viewer);CHKERRQ(ierr);
    ierr = VecView(U, viewer);CHKERRQ(ierr);
    
    // --------------------------------------------
    // Get the norms of errors (only for periodic
    // test cases)
    //---------------------------------------------

    PetscReal nrm_2, nrm_inf;
    ierr = ErrorNorms(U, da, Ctx, &nrm_2, &nrm_inf);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Norm2 = %.7e, NormMax = %.7e\n", nrm_2, nrm_inf);CHKERRQ(ierr);

    // --------------------------------------------
    // Free all the memory, finalize MPI and exit   
    //---------------------------------------------
    
    ierr = VecDestroy(&U);CHKERRQ(ierr);
    ierr = VecDestroy(&RHS);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = TSDestroy(&ts);CHKERRQ(ierr);
    ierr = VecDestroy(&Ctx.localU);CHKERRQ(ierr);
    
    free6d(Ctx.u_bnd);
    free4d(Ctx.F);
    free4d(Ctx.G);   
    free4d(Ctx.H);   
    
    ierr =  PetscTime(&end_time);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Time taken =  %g\n",(double)(end_time - start_time));CHKERRQ(ierr);
    
    ierr = PetscFinalize();CHKERRQ(ierr);
}
