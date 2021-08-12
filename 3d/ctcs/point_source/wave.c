static char help[] = "Solving 3d acoustic wave equation using finite difference with point source\n";


#include <petsc.h>


/*simulation parameters stored in a struct*/
typedef struct {

    PetscReal      xmin, xmax;    /*domain range [xmin, xmax]*/
    PetscReal      ymin, ymax;    /*domain range [ymin, ymax]*/
    PetscReal      zmin, zmax;    /*domain range [zmin, zmax]*/
    PetscInt       nx;            /*no of grid points along x direction*/
    PetscInt       ny;            /*no of grid points along y direction*/
    PetscInt       nz;            /*no of grid points along z direction*/
    PetscReal      dx;            /*grid spacing in x direction*/
    PetscReal      dy;            /*grid spacing in y direction*/
    PetscReal      dz;            /*grid spacing in z direction*/
    PetscInt       stencil;       /*stencil width*/
    
    PetscReal      c0;            /*wave speed*/
    
    PetscReal      dt;            /*time step*/
    PetscReal      nt;            /*no of time steps*/
    
    PetscReal      cfl;           /*cfl number*/

    PetscInt       isx;           /*source   location in x direction*/
    PetscInt       isy;           /*source   location in y direction*/
    PetscInt       isz;           /*source   location in z direction*/
    PetscInt       irx;           /*reciever location in x direction*/
    PetscInt       iry;           /*reciever location in y direction*/
    PetscInt       irz;           /*reciever location in z direction*/
    
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
    user.xmax           =   50.0;
    user.ymin           =   0.0;
    user.ymax           =   50.0;
    user.zmin           =   0.0;
    user.zmax           =   50.0;
    user.nx             =   100;
    user.ny             =   100;
    user.nz             =   100;
    user.dx             =   0.5;
    user.dy             =   0.5;
    user.dz             =   0.5;
    user.stencil        =   2;
    user.c0             =   250.0;
    user.dt             =   0.0005;
    user.nt             =   251;
    user.cfl            =   user.c0*user.dt/user.dx;

    user.isx            =   50;
    user.isy            =   50;
    user.isz            =   50;
    user.irx            =   57;
    user.iry            =   57;
    user.irz            =   57;

    ierr = PetscPrintf(PETSC_COMM_WORLD, "cfl = %g\n", user.cfl); CHKERRQ(ierr);

    /*create 3d distributed grid*/
    ierr = DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, user.nx, user.ny, user.nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, user.stencil, NULL, NULL, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, user.xmin, user.xmax, user.ymin, user.ymax, user.zmin, user.zmax); CHKERRQ(ierr);

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
        if (it % 5 == 0) ierr = write_vts(da, P, it); CHKERRQ(ierr);

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
    PetscInt        i, j, k, xs, xm, ys, ym, zs, zm;
    Vec             Plocal;
    PetscScalar     ***pold, ***p, ***pnew, ***c;
    PetscReal       pxx, pyy, pzz;
    PetscReal       dx = user->dx, dy = user->dy, dz = user->dz;
    PetscReal       invdx2 = 1.0/(dx*dx), invdy2 = 1.0/(dy*dy), invdz2 = 1.0/(dz*dz); 
    PetscReal       dt = user->dt;  

    /*get local grid boundaries*/
    ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);

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
    for (k = zs; k < zs + zm; ++k){
        for (j = ys; j < ys + ym; ++j){
            for (i = xs; i < xs + xm; ++i){

                /*compute pxx, pyy and pzz for 3 point and 5 point stencil*/
                switch (user->stencil)
                {
                case 1:
                    pxx = (p[k][j][i + 1] - 2.0 * p[k][j][i] + p[k][j][i - 1]) * invdx2;
                    pyy = (p[k][j + 1][i] - 2.0 * p[k][j][i] + p[k][j - 1][i]) * invdy2;
                    pzz = (p[k + 1][j][i] - 2.0 * p[k][j][i] + p[k - 1][j][i]) * invdz2;
                    break;
                case 2:
                    pxx = (-1.0 / 12 * p[k][j][i + 2] + 4.0 / 3.0 * p[k][j][i + 1] - 5.0 / 2.0 * p[k][j][i] + 4.0 / 3.0 * p[k][j][i - 1] - 1.0 / 12.0 * p[k][j][i - 2]) * invdx2;
                    pyy = (-1.0 / 12 * p[k][j + 2][i] + 4.0 / 3.0 * p[k][j + 1][i] - 5.0 / 2.0 * p[k][j][i] + 4.0 / 3.0 * p[k][j - 1][i] - 1.0 / 12.0 * p[k][j - 2][i]) * invdy2;
                    pzz = (-1.0 / 12 * p[k + 2][j][i] + 4.0 / 3.0 * p[k + 1][j][i] - 5.0 / 2.0 * p[k][j][i] + 4.0 / 3.0 * p[k - 1][j][i] - 1.0 / 12.0 * p[k - 2][j][i]) * invdz2;
                     
                    break;

                default:
                    SETERRQ(PETSC_COMM_WORLD, ierr, "stencil width can be either 1 or 2\n");
                }

                /*update pressure*/
                pnew[k][j][i] = 2.0*p[k][j][i] - pold[k][j][i] + c[k][j][i]*c[k][j][i]*dt*dt*(pxx + pyy + pzz);

                /*add source at (isx, isy, isz)*/
                if (i == user->isx && j == user->isy && k == user->isz) 
                    pnew[k][j][i] += Source(it*dt)/(dx*dy*dz) *dt*dt;
                
                /*print reciever presure*/
                if (i == user->irx && j == user->iry && k == user->irz){
                    FILE *f = fopen("results/p.dat","a+");
                    ierr = PetscFPrintf(PETSC_COMM_SELF, f, "%e\t%e\n", it*dt, p[k][j][i]); CHKERRQ(ierr);
                    fclose(f);
                }
            }
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

    PetscReal   f0 = 100.0;      /*dominant frequency is 40 Hz*/
    PetscReal   t0 = 4.0/ f0;   /*source time shift*/
    
    return -2.0*(t - t0)*f0*f0*PetscExpReal( -1.0*f0*f0*(t - t0)*(t - t0));
}

