static char help[] = "1d finite difference (CTCS scheme) acoustic solver for travelling wave in a periodic domain\n";

#include <petsc.h>


/*simulation parameters stored in a struct*/
typedef struct {

    PetscReal      xmin, xmax;    /*domain range [xmin, xmax]*/
    PetscInt       N;             /*no of grid points*/
    PetscReal      h;             /*grid spacing*/
    PetscReal      c0;            /*wave speed*/
    PetscReal      cfl;           /*cfl number*/
    PetscReal      dt;            /*time step*/
    PetscReal      tf;            /*final time*/
    PetscReal      c_disp;        /*dispersion veclocity*/

} AppCtx;


/*functions*/
PetscErrorCode ProjectFunction(DM, Vec, AppCtx*, PetscReal);
PetscErrorCode CTCS(DM, Vec, Vec, Vec, AppCtx*);
PetscErrorCode WriteSolution(DM, int, double, Vec, void *);
PetscErrorCode write_vts(DM, Vec, Vec, PetscInt, PetscReal);

PetscReal ExactSolution(PetscReal, PetscReal);
PetscReal PhaseVelocity(PetscReal, PetscReal, PetscReal, PetscReal);


int main(int argc, char **argv){

    /*petsc objects*/
    DM  da;
    Vec Uold, Ucurr, Unew, Uexact;

    /*set simulation parameters*/
    AppCtx  user;

    user.xmin   =   0.0;
    user.xmax   =   1.0;  
    user.N      =   100;
    user.h      =   (user.xmax - user.xmin)/(PetscReal)(user.N);
    user.c0     =   1.0;
    user.cfl    =   0.5;
    user.dt     =   user.cfl*user.h/user.c0;
    user.tf     =   1.0;


    /*time and step*/
    PetscReal t;
    PetscInt  step;

    /*initialize petsc program*/
    PetscErrorCode  ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

    /*phase velocity*/
    ierr = PetscPrintf(PETSC_COMM_WORLD, "wave speed = %g\n", user.c0); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "phase velocity = %g\n", PhaseVelocity(2*PETSC_PI, user.h, user.dt, user.c0)); CHKERRQ(ierr);


    /*create 1d distributed grid*/
    ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, user.N, 1, 1, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, user.xmin, user.xmax, 0, 0, 0, 0); CHKERRQ(ierr);

    /*create solution vectors*/
    ierr = DMCreateGlobalVector(da, &Uold); CHKERRQ(ierr);
    ierr = VecDuplicate(Uold, &Ucurr); CHKERRQ(ierr);
    ierr = VecDuplicate(Uold, &Unew); CHKERRQ(ierr);
    ierr = VecDuplicate(Uold, &Uexact); CHKERRQ(ierr);

    /*Name vectors*/
    ierr = PetscObjectSetName((PetscObject)Uexact, "Pexact");
    ierr = PetscObjectSetName((PetscObject)Unew, "P");


    /*initialize solution vector at t = 0*/
    t = 0.0;
    step = 0;
    ierr = ProjectFunction(da, Uold, &user, t); CHKERRQ(ierr);

    /*initialize solution vector at t + dt*/
    t += user.dt;
    step = 1;
    ierr = ProjectFunction(da, Ucurr, &user, t); CHKERRQ(ierr);

    /*evolve in time*/
    while (t < user.tf){

        t += user.dt;
        step++;

        /*update solution using CTCS scheme*/
        ierr = CTCS(da, Uold, Ucurr, Unew, &user); CHKERRQ(ierr);

        ierr = VecCopy(Ucurr, Uold); CHKERRQ(ierr);  /*Uold = Ucurr*/
        ierr = VecCopy(Unew , Ucurr); CHKERRQ(ierr); /*Ucurr = Unew*/

        /*get exact solution*/
        ierr = ProjectFunction(da, Uexact, &user, t); CHKERRQ(ierr);

        /*Plot numerical and exact solution*/
        if (step % 100 == 0) {
            ierr = write_vts(da, Unew, Uexact, step, t); CHKERRQ(ierr);
        }

    }

    /*Linfty norm*/
    PetscReal norm;
    ierr = VecNorm(Unew, NORM_INFINITY, &norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Linfty norm at t = %g is %e\n", t, norm); CHKERRQ(ierr);


    /*destroy*/
    DMDestroy(&da);
    VecDestroy(&Uold);
    VecDestroy(&Ucurr);
    VecDestroy(&Unew);
    VecDestroy(&Uexact);

    return PetscFinalize();
}

PetscReal ExactSolution(PetscReal x, PetscReal t){

    PetscReal k = 2*PETSC_PI;      /*wave number*/
    PetscReal c0 = 1.0;            /*wave speed*/

    /*dispersion relation*/
    PetscReal omega =  c0*k;       /*frequency*/         

    return PetscSinReal(k*x - omega*t);
}

PetscReal PhaseVelocity(PetscReal k, PetscReal h, PetscReal dt, PetscReal c){

    PetscReal cfl = c*dt/h;
    return ( 2.0/(k*dt) )*PetscAsinReal(cfl * PetscSinReal( k*h*0.5 ));
}


PetscErrorCode ProjectFunction(DM da, Vec U, AppCtx* user, PetscReal t){

    PetscErrorCode  ierr;
    PetscInt        i,xs,xm;
    PetscScalar     *u;
    PetscScalar     h = user->h;

    PetscFunctionBeginUser;

    /*get local grid boundaries*/
    ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL); CHKERRQ(ierr);

    /*get 1d array*/
    ierr = DMDAVecGetArray(da, U, &u); CHKERRQ(ierr);

    /*compute function over the locally owned part of the grid*/
    for (i = xs; i < xs + xm; ++i)
        u[i] = ExactSolution(i*h, t);

    /*restore array*/
    ierr = DMDAVecRestoreArray(da, U, &u); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode CTCS(DM da, Vec Uold, Vec Ucurr, Vec Unew, AppCtx* user){

    PetscErrorCode  ierr;
    PetscInt        i,xs,xm;
    Vec             Ulocal;
    PetscScalar     *uold;
    PetscScalar     *ucurr;
    PetscScalar     *unew;
    PetscScalar     cfl = user->cfl;


    PetscFunctionBeginUser;

    /*get local grid boundaries*/
    ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL); CHKERRQ(ierr);

    /*create local vector that has space for ghost values*/
    ierr = DMGetLocalVector(da, &Ulocal); CHKERRQ(ierr);

    /*scatter values from global to local vector*/
    ierr = DMGlobalToLocalBegin(da, Ucurr, INSERT_VALUES, Ulocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Ucurr, INSERT_VALUES, Ulocal); CHKERRQ(ierr);

    /*get array from vector*/
    ierr = DMDAVecGetArrayRead(da, Uold, &uold); CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da, Ulocal, &ucurr); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Unew, &unew); CHKERRQ(ierr);


    /*update locally owned solution using CTCS scheme*/
    for (i = xs; i < xs + xm; ++i)
        unew[i] = cfl*cfl*( ucurr[i+1] - 2.0*ucurr[i] + ucurr[i-1] ) + 2.0*ucurr[i] - uold[i];


    /*restore array*/
    ierr = DMDAVecRestoreArrayRead(da, Uold, &uold); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(da, Ulocal, &ucurr); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Unew, &unew); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &Ulocal); CHKERRQ(ierr);


    PetscFunctionReturn(0);
}

PetscErrorCode write_vts(DM da, Vec U, Vec Uexact, PetscInt step, PetscReal time)
{
    PetscErrorCode  ierr;
    PetscViewer viewer;
    char filename[20];

    sprintf(filename, "sol-%08d.vts", step);
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr); 
    ierr = DMView(da, viewer); CHKERRQ(ierr);
    ierr = VecView(U, viewer); CHKERRQ(ierr);
    ierr = VecView(Uexact, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    return 0;
}
