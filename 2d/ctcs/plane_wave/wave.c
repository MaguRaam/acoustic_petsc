static char help[] = "Solving 2d acoustic wave equation using finite difference CTCS sheme in a periodic domain\n";

#include <petsc.h>

/*simulation parameters stored in a struct*/
typedef struct {

    PetscReal      xmin, xmax;    /*domain range [xmin, xmax]*/
    PetscReal      ymin, ymax;    /*domain range [ymin, ymax]*/
    PetscInt       Nx;            /*no of grid points along x direction*/
    PetscInt       Ny;            /*no of grid points along y direction*/
    PetscReal      h;             /*grid spacing*/
    PetscReal      c0;            /*wave speed*/
    PetscReal      cfl;           /*cfl number*/
    PetscReal      dt;            /*time step*/
    PetscReal      tf;            /*final time*/

} AppCtx;


/*functions*/
PetscErrorCode ProjectFunction(DM, Vec, AppCtx*, PetscReal);
double ExactSolution(PetscReal, PetscReal, PetscReal);
PetscErrorCode CTCS(DM, Vec, Vec, Vec, AppCtx*);
PetscErrorCode write_vts(DM, Vec, Vec, PetscInt, PetscReal);


int main(int argc, char** argv){

    /*petsc objects*/
    DM          da;
    Vec         Pold, Pcurr, Pnew, Pexact;

    /*set simulation parameters*/
    AppCtx      user;
    user.xmin   =    0.0;
    user.xmax   =    1.0;
    user.ymin   =    0.0;
    user.ymax   =    1.0;
    user.Nx     =    64;
    user.Ny     =    64;
    user.h      =    (user.xmax - user.xmin)/(PetscReal)(user.Nx);
    user.c0     =    1.0;
    user.cfl    =    0.5;
    user.dt     =    0.5*user.cfl*user.h/user.c0;
    user.tf     =    100;

    /*initialize petsc program*/
    PetscErrorCode  ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

    /*create 2d distributed grid*/
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, user.Nx, user.Ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, user.xmin, user.xmax, user.ymin, user.ymax, 0, 0); CHKERRQ(ierr);

    /*create solution vectors*/
    ierr = DMCreateGlobalVector(da, &Pold); CHKERRQ(ierr);
    ierr = VecDuplicate(Pold, &Pcurr); CHKERRQ(ierr);
    ierr = VecDuplicate(Pold, &Pnew); CHKERRQ(ierr);
    ierr = VecDuplicate(Pold, &Pexact); CHKERRQ(ierr);

    /*Name vectors*/
    ierr = PetscObjectSetName((PetscObject)Pexact, "Pexact");
    ierr = PetscObjectSetName((PetscObject)Pnew, "P");

    /*initialize solution vector at t = 0*/
    PetscReal t = 0.0;
    PetscInt step = 0;
    ierr = ProjectFunction(da, Pold, &user, t); CHKERRQ(ierr);

    /*initialize solution vector at t + dt*/
    t += user.dt;
    step = 1;
    ierr = ProjectFunction(da, Pcurr, &user, t); CHKERRQ(ierr);

    /*evolve in time*/
    while (t < user.tf){

        t += user.dt;
        step++;

        /*update solution using CTCS scheme*/
        ierr = CTCS(da, Pold, Pcurr, Pnew, &user); CHKERRQ(ierr);

        ierr = VecCopy(Pcurr, Pold); CHKERRQ(ierr);  /*Pold = Pcurr*/
        ierr = VecCopy(Pnew , Pcurr); CHKERRQ(ierr); /*Pcurr = Pnew*/

        /*get exact solution*/
        ierr = ProjectFunction(da, Pexact, &user, t); CHKERRQ(ierr);

        /*Plot numerical and exact solution*/
        if (step % 1000 == 0) {
            ierr = write_vts(da, Pnew, Pexact, step, t); CHKERRQ(ierr);

            /*Linfty norm*/
            PetscReal norm;
            ierr = VecNorm(Pnew, NORM_INFINITY, &norm); CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD, "Linfty norm at t = %g is %e\n", t, norm); CHKERRQ(ierr);

        }

    }


    /*destroy*/
    DMDestroy(&da);
    VecDestroy(&Pold);
    VecDestroy(&Pcurr);
    VecDestroy(&Pnew);
    VecDestroy(&Pexact);


    return PetscFinalize();
}



double ExactSolution(PetscReal x, PetscReal y, PetscReal t){               
 
    PetscReal kx = 2*PETSC_PI;
    PetscReal ky = 0;
    PetscReal c0 = 1.0;

    //dispersion relation:
    PetscReal omega = c0*sqrt(kx*kx + ky*ky);

    return sin(kx*x + ky*y - omega*t);
} 


PetscErrorCode ProjectFunction(DM da, Vec P, AppCtx* user, PetscReal t){

    PetscErrorCode  ierr;
    PetscInt        i,j,xs,ys,xm,ym;
    PetscScalar     **p;
    PetscScalar     x, y, h = user->h;

    PetscFunctionBeginUser;

    /*get local grid boundaries*/
    ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL); CHKERRQ(ierr);

    /*get 2d array*/
    ierr = DMDAVecGetArray(da, P, &p); CHKERRQ(ierr);

    /*compute function over the locally owned part of a grid*/
    for (j = ys; j < ys + ym; ++j){
        y = j*h;
        for (i = xs; i < xs + xm; ++i){
            x = i*h;
            p[j][i] = ExactSolution(x, y, t);
        }
    }

    ierr = DMDAVecRestoreArray(da, P, &p); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


PetscErrorCode CTCS(DM da, Vec Pold, Vec Pcurr, Vec Pnew, AppCtx* user){

    PetscErrorCode  ierr;
    PetscInt        i, j, xs, ys, xm, ym;
    Vec             Plocal;
    PetscScalar     **pold;
    PetscScalar     **pcurr;
    PetscScalar     **pnew;
    PetscScalar     alpha = (user->c0) * (user->dt)/(user->h);


    PetscFunctionBeginUser;

    /*get local grid boundaries*/
    ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL); CHKERRQ(ierr);

    /*create local vector that has space for ghost values*/
    ierr = DMGetLocalVector(da, &Plocal); CHKERRQ(ierr);

    /*scatter values from global to local vector*/
    ierr = DMGlobalToLocalBegin(da, Pcurr, INSERT_VALUES, Plocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Pcurr, INSERT_VALUES, Plocal); CHKERRQ(ierr);

    /*get array from vector*/
    ierr = DMDAVecGetArrayRead(da, Pold, &pold); CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da, Plocal, &pcurr); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Pnew, &pnew); CHKERRQ(ierr);

    /*update locally owned solution using CTCS scheme*/
    for (j = ys; j < ys + ym; ++j)
        for (i = xs; i < xs + xm; ++i)
            pnew[j][i] = alpha*alpha*( pcurr[j+1][i] + pcurr[j-1][i] + pcurr[j][i+1] + pcurr[j][i-1] - 4.0*pcurr[j][i] ) + 2.0*pcurr[j][i] - pold[j][i];


    /*restore array*/
    ierr = DMDAVecRestoreArrayRead(da, Pold, &pold); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(da, Plocal, &pcurr); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Pnew, &pnew); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &Plocal); CHKERRQ(ierr);

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