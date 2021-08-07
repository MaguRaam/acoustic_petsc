static char help[] = "1d finite difference acoustic solver with delta function source, the solution is a Greens Function\n";

#include <petsc.h>

/* 1D Green's function

    ∂2tG(x,t;x0,t0) − c2ΔG(x,t;x0,t0) =δ(x−x0)δ(t−t0)

    The delta function source is located at x0 and t0, Therefore the Green function solution looks like,

    G= (1/2c) H(t - t0 − |x -x0|/c)
*/

/*simulation parameters stored in a struct*/
typedef struct {

    PetscReal   xmin, xmax;    /*domain range*/
    PetscInt    nx;            /*no of grid pts*/
    PetscReal   dx;            /*grid spacing*/
    PetscInt    stencil;       /*stencil width*/

    PetscReal   c0;            /*wave speed*/
    
    PetscReal   dt;            /*time step*/
    PetscReal   nt;            /*no of time steps*/

    PetscReal   cfl;           /*cfl*/

    PetscInt    isrc;          /*source location*/
    PetscInt    ir;            /*reciever location*/

} AppCtx;


/*functions to solve pde*/
PetscErrorCode UpdatePressure(DM, Vec, Vec, Vec, Vec, AppCtx*, PetscInt);
PetscErrorCode write_vts(DM, Vec, Vec, PetscInt);
PetscReal  GreensFunction(PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
PetscErrorCode  ComputeGreensFunction(DM, Vec, AppCtx*, PetscReal);




int main(int argc, char **argv){

    /*initialize petsc program*/
    PetscErrorCode  ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);


    /*petsc objects*/
    DM          da;                         /*grid object*/
    Vec         Pold, P, Pnew;              /*pressure at n-1, n and n+1*/
    Vec         C;                          /*wave speed*/
    Vec         Green;                      /**/

    /*set simulation parameters*/
    AppCtx      user;

    user.xmin       =   0.0;
    user.xmax       =   1.0;
    user.nx         =   1000;   
    user.dx         =   ( user.xmax - user.xmin )/ (PetscReal)user.nx;
    user.stencil    =   2;
    user.c0         =   1.0;  
    user.dt         =   0.0001;
    user.nt         =   10000;
    user.cfl        =   user.c0*user.dt/user.dx;
    user.isrc       =   500;
    user.ir         =   750;

    ierr = PetscPrintf(PETSC_COMM_WORLD, "cfl no = %g\n", user.cfl); CHKERRQ(ierr);

    /*create 1d distributed grid*/
    ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, user.nx, 1, user.stencil, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, user.xmin, user.xmax, 0, 0, 0, 0); CHKERRQ(ierr);

    /*create solution vectors*/
    ierr = DMCreateGlobalVector(da, &P); CHKERRQ(ierr);
    ierr = VecDuplicate(P, &Pold); CHKERRQ(ierr);
    ierr = VecDuplicate(P, &Pnew); CHKERRQ(ierr);
    ierr = VecDuplicate(P, &C); CHKERRQ(ierr);
    ierr = VecDuplicate(P, &Green); CHKERRQ(ierr);

    /*name pressure vector useful while plotting*/
    ierr = PetscObjectSetName((PetscObject)P, "P"); 
    ierr = PetscObjectSetName((PetscObject)Green, "Green");

    /*initialize pressure vectors to zeros*/
    ierr = VecSet(P, 0.0); CHKERRQ(ierr);
    ierr = VecSet(Pold, 0.0); CHKERRQ(ierr);
    ierr = VecSet(Pnew, 0.0); CHKERRQ(ierr);

    /*set homogeneous wave speed c0*/
    ierr = VecSet(C, user.c0); CHKERRQ(ierr); 

    /*evolve in time*/
    for (int it = 0; it < user.nt; ++it){

        /*update pressure using Leap Frog scheme*/
        ierr = UpdatePressure(da, Pold, P, Pnew, C, &user, it); CHKERRQ(ierr);

        /*remap time levels*/
        ierr = VecCopy(P, Pold); CHKERRQ(ierr);  /*Pold = P*/
        ierr = VecCopy(Pnew , P); CHKERRQ(ierr); /*P = Pnew*/

        /*plot pressure*/
        if (it % 500== 0) {

            ierr = ComputeGreensFunction(da, Green, &user, it*user.dt); CHKERRQ(ierr);
            ierr = write_vts(da, P, Green, it); CHKERRQ(ierr);
        }

    }

    /*destroy objects*/
    DMDestroy(&da);
    VecDestroy(&P);
    VecDestroy(&Pold);
    VecDestroy(&Pnew);
    VecDestroy(&C);                                         
    VecDestroy(&Green);

    return PetscFinalize();
}




PetscErrorCode UpdatePressure(DM da, Vec Pold, Vec P, Vec Pnew, Vec C, AppCtx* user, PetscInt it){

    PetscErrorCode  ierr;
    PetscInt        i,xs,xm;
    Vec             Plocal;
    PetscScalar     *pold;
    PetscScalar     *p;
    PetscScalar     *pnew;
    PetscScalar     *c;
    PetscReal       pxx;
    PetscReal       invdx2 = 1.0/ (user->dx*user->dx);
    PetscReal       invdx  = 1.0/ user->dx; 
    PetscReal       dt = user->dt;

    PetscFunctionBeginUser;

    /*get local grid boundaries*/
    ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL); CHKERRQ(ierr);

    /*create local vector that has space for ghost values*/
    ierr = DMGetLocalVector(da, &Plocal); CHKERRQ(ierr);

    /*scatter values from global to local vector*/
    ierr = DMGlobalToLocalBegin(da, P, INSERT_VALUES, Plocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,   P, INSERT_VALUES, Plocal); CHKERRQ(ierr);

    /*get array from vector*/
    ierr = DMDAVecGetArrayRead(da, Pold, &pold); CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da, Plocal, &p); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Pnew, &pnew); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, C, &c); CHKERRQ(ierr);

    /*update locally owned solution using CTCS scheme*/
    for (i = xs; i < xs + xm; ++i){

        /*compute pxx*/
        switch (user->stencil)
        {PetscFunctionReturn(0);
        case 1:
            pxx = (p[i+1] - 2.0*p[i] + p[i-1])*invdx2;

            break;
        case 2:
            pxx = ( -1.0/12 * p[i + 2] + 4.0/3.0  * p[i + 1] - 5.0/2.0 * p[i] + 4.0/3.0  * p[i - 1] - 1.0/12.0 * p[i - 2])*invdx2;
            break;

        default:
             SETERRQ(PETSC_COMM_WORLD,ierr,"stencil width can be either 1 or 2\n");
        }

        /*update pressure*/
        pnew[i] = 2.0*p[i] - pold[i] + c[i]*c[i]*dt*dt*pxx;

        /*put source at isrc*/
        if (it == 0 && i == user->isrc) 
            pnew[i] += invdx*dt*dt;
    }

    


    /*restore array*/ 
    ierr = DMDAVecRestoreArrayRead(da, Pold, &pold); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(da, Plocal, &p); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Pnew, &pnew); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, C, &c); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &Plocal); CHKERRQ(ierr);


    PetscFunctionReturn(0);
}

PetscErrorCode write_vts(DM da, Vec P, Vec Green, PetscInt step)
{
    PetscErrorCode  ierr;
    PetscViewer viewer;
    char filename[20];

    sprintf(filename, "sol-%08d.vts", step);
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr); 
    ierr = DMView(da, viewer); CHKERRQ(ierr);
    ierr = VecView(P, viewer); CHKERRQ(ierr);
    ierr = VecView(Green, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    return 0;
}

PetscReal  GreensFunction(PetscReal x, PetscReal x0, PetscReal t, PetscReal t0, PetscReal c0){

    return ( (t - t0 - PetscAbsReal(x - x0)/c0) >= 0.0 ) ? 0.5*c0 : 0.0;
}

PetscErrorCode  ComputeGreensFunction(DM da, Vec Green, AppCtx* user, PetscReal t){
    
    PetscErrorCode  ierr;
    PetscInt        i,xs,xm;
    PetscScalar     *green;
    PetscScalar     dx = user->dx;
    PetscInt        isrc = user->isrc;

    PetscFunctionBeginUser;

    /*get local grid boundaries*/
    ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL); CHKERRQ(ierr);

    /*get array from vector*/
    ierr = DMDAVecGetArray(da, Green, &green); CHKERRQ(ierr);

    /*compute Greens function over locally owned part of grid*/
    for (i = xs; i < xs + xm; ++i)
        green[i] = GreensFunction(i*dx, isrc*dx, t, 0.0, user->c0); 

    /*restore array*/ 
    ierr = DMDAVecRestoreArray(da, Green, &green); CHKERRQ(ierr); 

    PetscFunctionReturn(0);
}   
