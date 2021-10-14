/*
 * dissipation.c
 *      Author: sunder
 */ 
#include "hype.h" 

//----------------------------------------------------------------------------
// Find the rate of dissipation. 
//----------------------------------------------------------------------------

PetscErrorCode Dissipation(Vec U, DM da, AppCtx *Ctx, PetscReal* dr) {
    
    PetscErrorCode ierr; 
    Field   ***u;
    PetscReal dr_local = 0.0, r1_2h = 0.5/Ctx->h; 
    PetscReal u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z, tau_xx, tau_xy, tau_yy, tau_yz, tau_xz, tau_zz, mu, div_v; 
    PetscInt i,j,k,xs,ys,zs,xm,ym,zm;
    
    // Scatter global->local to have access to the required ghost values 

    ierr = DMGlobalToLocalBegin(da, U, INSERT_VALUES, Ctx->localU);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, U, INSERT_VALUES,   Ctx->localU);CHKERRQ(ierr);

    // Read the local solution to the array u  

    ierr = DMDAVecGetArrayRead(da, Ctx->localU, &u); CHKERRQ(ierr); 
    
    ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
    
    
    mu = MU_0; 
    
    
    for ( k = zs; k < zs+zm; ++k ) {
        for ( j = ys; j < ys+ym; ++j ) { 
            for ( i = xs; i < xs+xm; ++i ) {
                
            
                u_x = r1_2h*(u[k][j][i+1].comp[1]/u[k][j][i+1].comp[0] - u[k][j][i-1].comp[1]/u[k][j][i-1].comp[0]);
                u_y = r1_2h*(u[k][j+1][i].comp[1]/u[k][j+1][i].comp[0] - u[k][j-1][i].comp[1]/u[k][j-1][i].comp[0]);
                u_z = r1_2h*(u[k+1][j][i].comp[1]/u[k+1][j][i].comp[0] - u[k-1][j][i].comp[1]/u[k-1][j][i].comp[0]);
                
                v_x = r1_2h*(u[k][j][i+1].comp[2]/u[k][j][i+1].comp[0] - u[k][j][i-1].comp[2]/u[k][j][i-1].comp[0]);
                v_y = r1_2h*(u[k][j+1][i].comp[2]/u[k][j+1][i].comp[0] - u[k][j-1][i].comp[2]/u[k][j-1][i].comp[0]);
                v_z = r1_2h*(u[k+1][j][i].comp[2]/u[k+1][j][i].comp[0] - u[k-1][j][i].comp[2]/u[k-1][j][i].comp[0]);
                
                w_x = r1_2h*(u[k][j][i+1].comp[3]/u[k][j][i+1].comp[0] - u[k][j][i-1].comp[3]/u[k][j][i-1].comp[0]);
                w_y = r1_2h*(u[k][j+1][i].comp[3]/u[k][j+1][i].comp[0] - u[k][j-1][i].comp[3]/u[k][j-1][i].comp[0]);
                w_z = r1_2h*(u[k+1][j][i].comp[3]/u[k+1][j][i].comp[0] - u[k-1][j][i].comp[3]/u[k-1][j][i].comp[0]);
                
                div_v = -(2.0/3.0)*mu*(u_x + v_y + w_z);
                
                tau_xx = div_v + 2.0*mu*u_x;
                tau_xy = mu*(u_y + v_x) ;
                tau_xz = mu*(u_z + w_x) ;
                tau_yy = div_v + 2.0*mu*v_y;
                tau_yz = mu*(v_z + w_y) ;
                tau_zz = div_v + 2.0*mu*w_z;
                
                dr_local += (tau_xx*u_x + tau_xy*u_y + tau_xz*u_z) + (tau_xy*v_x + tau_yy*v_y + tau_yz*v_z) + (tau_xz*w_x + tau_yz*w_y + tau_zz*w_z); 
                
                
            }
        }
    }
    
    ierr = MPI_Allreduce(&dr_local, dr, 1, MPIU_REAL,MPIU_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(da,Ctx->localU,&u);CHKERRQ(ierr);

    return ierr; 
}
     
