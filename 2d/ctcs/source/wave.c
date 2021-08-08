static char help[] = "Solving 2d acoustic wave equation using finite difference with point source\n";


#include <petsc.h>


/*simulation parameters stored in a struct*/
typedef struct {

    PetscReal      xmin, xmax;    /*domain range [xmin, xmax]*/
    PetscReal      ymin, ymax;    /*domain range [ymin, ymax]*/
    PetscInt       nx;            /*no of grid points along x direction*/
    PetscInt       ny;            /*no of grid points along y direction*/
    PetscReal      dx;            /*grid spacing in x direction*/
    PetscReal      dy;            /*grid spacing in y direction*/
    PetscInt       stencil;       /*stencil width*/
    
    PetscReal      c0;            /*wave speed*/
    
    PetscReal      dt;            /*time step*/
    PetscReal      nt;            /*no of time steps*/
    
    PetscReal      cfl;           /*cfl number*/

    PetscInt       isx;           /*source   location in x direction*/
    PetscInt       isy;           /*source   location in y direction*/
    PetscInt       irx;           /*reciever location in x direction*/
    PetscInt       iry;           /*reciever location in y direction*/
    
} AppCtx;


/*function declaration*/
PetscErrorCode UpdatePressure(DM, Vec, Vec, Vec, Vec, AppCtx*, PetscInt);
PetscErrorCode write_vts(DM, Vec, PetscInt);
PetscReal Source(PetscReal);


int main(int argc, char **argv){

    /*initialize petsc program*/
    PetscErrorCode  ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

    /*petsc objects*/
    DM          da;
    Vec         Pold, P, Pnew; 
    Vec         C;

    /*set simulation parameters*/
    AppCtx      user;

    user.xmin           =   0.0;
    user.xmax           =   500;
    user.ymin           =   0.0;
    user.ymax           =   500;
    user.nx             =   500;
    user.ny             =   500;
    user.dx             =   1.0;
    user.dy             =   1.0;
    user.stencil        =   2;
    user.c0             =   580.0;
    user.dt             =   0.0010;
    user.nt             =   502;
    user.cfl            =   user.c0*user.dt/user.dx;

    user.isx            =   250;
    user.isy            =   250;
    user.irx            =   330;
    user.iry            =   330;

    ierr = PetscPrintf(PETSC_COMM_WORLD, "cfl = %g\n", user.cfl); CHKERRQ(ierr);

    /*create 2d distributed grid*/
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, user.nx, user.ny, PETSC_DECIDE, PETSC_DECIDE, 1, user.stencil, NULL, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, user.xmin, user.xmax, user.ymin, user.ymax, 0, 0); CHKERRQ(ierr);

    /*create solution vectors*/
    ierr = DMCreateGlobalVector(da, &P); CHKERRQ(ierr);
    ierr = VecDuplicate(P, &Pold); CHKERRQ(ierr);
    ierr = VecDuplicate(P, &Pnew); CHKERRQ(ierr);
    ierr = VecDuplicate(P, &C); CHKERRQ(ierr);

    /*name pressure vector useful while plotting*/
    ierr = PetscObjectSetName((PetscObject)P, "P"); 

    /*initialize pressure*/
    ierr = VecSet(P, 0.0); CHKERRQ(ierr);
    ierr = VecSet(Pold, 0.0); CHKERRQ(ierr);
    ierr = VecSet(Pnew, 0.0); CHKERRQ(ierr);

    /*set homogeneous wave speed*/
    ierr = VecSet(C, user.c0); CHKERRQ(ierr);


    /*evolve in time*/
    for (int it = 0; it < user.nt; ++it){

        /*update pressure using Leap Frog scheme*/
        ierr = UpdatePressure(da, Pold, P, Pnew, C, &user, it); CHKERRQ(ierr);

        /*remap time levels*/
        ierr = VecCopy(P, Pold); CHKERRQ(ierr);  /*Pold = P*/
        ierr = VecCopy(Pnew , P); CHKERRQ(ierr); /*P = Pnew*/

        /*plot pressure*/
        if (it % 50 == 0) ierr = write_vts(da, P, it); CHKERRQ(ierr);

    }




    /*destroy petsc objects*/
    DMDestroy(&da);
    VecDestroy(&P);
    VecDestroy(&Pold);
    VecDestroy(&Pnew);
    VecDestroy(&C); 


    return PetscFinalize();
}

/*function to update pressure using Leap Frog scheme*/
PetscErrorCode UpdatePressure(DM da, Vec Pold, Vec P, Vec Pnew, Vec C, AppCtx* user, PetscInt it){

    PetscErrorCode  ierr;
    PetscInt        i, j, xs, xm, ys, ym;
    Vec             Plocal;
    PetscScalar     **pold, **p, **pnew, **c;
    PetscReal       pxx, pyy;
    PetscReal       dx = user->dx, dy = user->dy;
    PetscReal       invdx2 = 1.0/(dx*dx), invdy2 = 1.0/(dy*dy); 
    PetscReal       dt = user->dt;

    PetscFunctionBeginUser;

    /*get local grid boundaries*/
    ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL); CHKERRQ(ierr);

    /*create local vector that has space for ghost values*/
    ierr = DMGetLocalVector(da, &Plocal); CHKERRQ(ierr);

    /*scatter values from global to local vector*/
    ierr = DMGlobalToLocalBegin(da, P, INSERT_VALUES, Plocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, P, INSERT_VALUES, Plocal); CHKERRQ(ierr);

    /*get array from vector*/
    ierr = DMDAVecGetArrayRead(da, Pold, &pold); CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da, Plocal, &p); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Pnew, &pnew); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, C, &c); CHKERRQ(ierr);

    /*update locally owned solution using CTCS scheme*/
    for (j = ys; j < ys + ym; ++j){
        for (i = xs; i < xs + xm; ++i){

            /*compute pxx and pyy for 3 point and 5 point stencil*/
            switch (user->stencil)
            {
            case 1:
                 pxx = (p[j][i+1] - 2.0*p[j][i] + p[j][i-1])*invdx2;
                 pyy = (p[j+1][i] - 2.0*p[j][i] + p[j-1][i])*invdy2;
                break;
            case 2:
                 pxx = ( -1.0/12 * p[j][i + 2] + 4.0/3.0  * p[j][i + 1] - 5.0/2.0 * p[j][i] + 4.0/3.0  * p[j][i - 1] - 1.0/12.0 * p[j][i - 2])*invdx2;
                 pyy = ( -1.0/12 * p[j + 2][i] + 4.0/3.0  * p[j + 1][i] - 5.0/2.0 * p[j][i] + 4.0/3.0  * p[j - 1][i] - 1.0/12.0 * p[j - 2][i])*invdy2;
                break;

            default:
                SETERRQ(PETSC_COMM_WORLD,ierr,"stencil width can be either 1 or 2\n");
            }

            /*update pressure*/
            pnew[j][i] = 2.0*p[j][i] - pold[j][i] + c[j][i]*c[j][i]*dt*dt*(pxx + pyy);

            /*put source at isrc*/
            if (i == user->isx && j == user->isy) 
                pnew[j][i] += Source(it*dt)/(dx*dy) *dt*dt;
        }
    }


    /*restore array*/
    ierr = DMDAVecRestoreArrayRead(da, Pold, &pold); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(da, Plocal, &p); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Pnew, &pnew); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, C, &c); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &Plocal); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


/*write solution vector in vts format*/
PetscErrorCode write_vts(DM da, Vec P, PetscInt step)
{
    PetscErrorCode  ierr;
    PetscViewer viewer;
    char filename[20];

    sprintf(filename, "sol-%08d.vts", step);
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr); 
    ierr = DMView(da, viewer); CHKERRQ(ierr);
    ierr = VecView(P, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    return 0;
}

PetscReal Source(PetscReal  t){

    PetscReal   f0 = 65.0;      /*dominant frequency is 40 Hz*/
    PetscReal   t0 = 4.0/ f0;   /*source time shift*/
    
    return -2.0*(t - t0)*f0*f0*PetscExpReal( -1.0*f0*f0*(t - t0)*(t - t0));
}

