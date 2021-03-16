static char help[] = "Solve wave equation using CTCS scheme\n";

#include <petsc.h>

typedef struct
{
    PetscInt    N;
    PetscReal   L;
    PetscReal   h;

    PetscReal   c;      /*wave speed*/
    PetscReal   cfl;    /*cfl no*/

    PetscReal   dt;     /*time step*/
    PetscReal   tf;     /*final time*/
} AppCtx;

int main(int argc, char** argv)
{
    /*initialize petsc program*/
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc,&argv,NULL,help); CHKERRQ(ierr);

    /*setup simulation parameters*/

    /*grid info*/
    AppCtx user;
    user.N = 1000;
    user.L = 2.0*PETSC_PI;
    user.h = user.L/(PetscReal)(user.N-1);

    /*wave speed and cfl*/
    user.c    =  1.0;
    user.cfl  =  0.5;

    /*time*/
    user.dt   =  (0.5*user.cfl*user.h)/user.c; 
    user.tf   =  2.0*PETSC_PI;

    /*create distributed structured grid with 1 ghost layer*/
    DM da;
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_STAR,user.N,user.N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);

    /*set coordinate*/
    ierr = DMDASetUniformCoordinates(da,0,user.L,0,user.L,0,0); CHKERRQ(ierr);

    /*Get local grid info: useful while looping over locally owned grid points*/
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);


    /*create petsc vector*/
    Vec uold, u, unew, source;
    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
    ierr = VecDuplicate(u,&uold); CHKERRQ(ierr);
    ierr = VecDuplicate(u,&unew); CHKERRQ(ierr);

    /*create petsc local vector for u that holds room for ghost values*/
    Vec ulocal;
    ierr = DMCreateLocalVector(da,&ulocal); CHKERRQ(ierr);

    /*create 2d array*/
    PetscReal **auold, **au, **aunew;

    /*initial condition*/
    ierr = VecSet(u,0.0); CHKERRQ(ierr);
    ierr = VecSet(uold,0.0); CHKERRQ(ierr);
    ierr = VecSet(unew,0.0); CHKERRQ(ierr);

    /*evolve*/
    PetscReal t = 0.0;
    PetscInt  step = 0;

    while (t < user.tf)
    {
        /*Get array from vector*/
        ierr = DMDAVecGetArrayRead(da,uold,&auold); CHKERRQ(ierr);
        ierr = DMDAVecGetArrayRead(da,unew,&aunew); CHKERRQ(ierr);
        ierr = DMGlobalToLocalBegin(da,u,INSERT_VALUES,ulocal); CHKERRQ(ierr);
        ierr = DMGlobalToLocalEnd(da,u,INSERT_VALUES,ulocal); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da,ulocal,&au); CHKERRQ(ierr);

        /*loop over locally owned grid points and update the solution*/
        for (int j = info.ys; j < info.ys + info.ym; ++j)
        {
            for (int i = info.xs; i < info.xs + info.xm; ++i)
            {
                //!set boundary time dependent dirchlet boundary condition at left boundary
                //if (i-1 == -1) au[j][-1] = 1.0;

                aunew[j][i] = (user.cfl)*(user.cfl)*(au[j+1][i] + au[j-1][i] + au[j][i+1] + au[j][i-1] - 4.0*au[j][i]) + 2.0*au[j][i] - auold[j][i];     
                if (t == 0.0)aunew[ (int)(info.ys + 0.5*info.ym)     ][ (int) (info.xs + 0.5*info.xm)  ] += 1.0; 
            }
        }

        /*Restore array from vector*/
        ierr = DMDAVecRestoreArrayRead(da,uold,&auold); CHKERRQ(ierr);
        ierr = DMDAVecRestoreArrayRead(da,unew,&aunew); CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(da,ulocal,&au); CHKERRQ(ierr);

        /*Copy vector*/
        ierr = VecCopy(u,uold); CHKERRQ(ierr);
        ierr = VecCopy(unew,u); CHKERRQ(ierr);

        /*increment time and time step*/
        t += user.dt;
        step++;
        PetscPrintf(PETSC_COMM_WORLD,"step, t = %d, %f\n", step, t);

        if ((step % 10) == 0)
        {
            /*write vts*/
            char filename[20];
            sprintf(filename, "plot/sol-%08d.vts", step); // 8 is the padding level, increase it for longer simulations 
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data in vts format to %s at t = %f, step = %d\n", filename, t, step);CHKERRQ(ierr);
            PetscViewer viewer;  
            ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr); 
            ierr = DMView(da, viewer);
            VecView(u, viewer);
            ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        }

    }



    /*petsc destroy*/
    DMDestroy(&da);
    VecDestroy(&u);
    VecDestroy(&uold);
    VecDestroy(&unew);

    return PetscFinalize();
}
