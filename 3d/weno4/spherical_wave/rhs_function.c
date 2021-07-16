/*
 * rhs_function.c
 *      Author: sunder
 */ 
#include "hype.h" 


//----------------------------------------------------------------------------
// Compute the value of RHS for each cell in the domain
//----------------------------------------------------------------------------

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec RHS, void* ctx) {

    PetscErrorCode ierr;           
    AppCtx *Ctx = (AppCtx*)ctx; 
    DM da;                         
    PetscInt i,j,k,f,xs,ys,zs,xm,ym,zm,xsg,ysg,zsg,xmg,ymg,zmg,oned_begin, oned_end,irhs,iDim;         
    
    Field   ***u;                   
    Field   ***rhs;                 
    PetscReal grad_x, grad_y, grad_z; 
    PetscReal u_x_loc[5], u_y_loc[5], u_z_loc[5], u_xy_loc[5], u_yz_loc[5], u_zx_loc[5], u_xyz_loc[8]; 
    PetscReal u_coeffs[dofs_per_cell];
    PetscReal r1_h = 1./(Ctx->h); 
    PetscReal nx, ny, nz; 
    PetscInt local_i, local_j, local_k;

    PetscReal grad_QL[nVar][DIM]; PetscReal grad_QR[nVar][DIM]; 
    PetscReal Flux, F;
    
    ierr = TSGetDM(ts, &da); CHKERRQ(ierr);

    // Scatter global->local to have access to the required ghost values 
    ierr = DMGlobalToLocalBegin(da, U, INSERT_VALUES, Ctx->localU);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, U, INSERT_VALUES,   Ctx->localU);CHKERRQ(ierr);

    // Read the local solution to the array u
    ierr = DMDAVecGetArrayRead(da, Ctx->localU, &u); CHKERRQ(ierr); 
    ierr = DMDAVecGetArray(da, RHS,&rhs);CHKERRQ(ierr);

    ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
    ierr = DMDAGetGhostCorners(da, &xsg, &ysg, &zsg, &xmg, &ymg, &zmg);

    //--------------------------------------------------------------
    // Apply Boundary Conditions 
    //--------------------------------------------------------------

    oned_begin = 0; oned_end = Ctx->N_x-1;  // First in the x-direction

    for (k = zsg; k < zsg+zmg; ++k) {
        for (j = ysg; j < ysg+ymg; ++j) {
            for (i = xsg; i < xsg+xmg; ++i) {
                
                if (i < 0)  { // Left boundary 
                    
                    // Adiabatic Wall Boundary

                    if (Ctx->left_boundary == adiabatic) {  
                        irhs = oned_begin - 1 - i; 
                        u[k][j][i].comp[0] = -u[k][j][irhs].comp[0];
                    }
                }
                
                if (i >= Ctx->N_x) { // Right Boundary 
                    
                    // Adiabatic Wall Boundary

                    if (Ctx->right_boundary == adiabatic) { 
                        irhs = 2*oned_end - i + 1; 
                        u[k][j][i].comp[0] = -u[k][j][irhs].comp[0];
                    }
                }
            
            }
        }
    }

    oned_begin = 0; oned_end = Ctx->N_y-1; // Next in the y-direction 

    for (k = zsg; k < zsg+zmg; ++k) {
        for (j = ysg; j < ysg+ymg; ++j) {
            for (i = xsg; i < xsg+xmg; ++i) {
                
                if (j < 0) { // Bottom boundary 
                    

                    // Adiabatic Boundary
                    if (Ctx->bottom_boundary == adiabatic) { 
                        irhs = oned_begin - 1 - j;
                        u[k][j][i].comp[0] = -u[k][irhs][i].comp[0]; 
                    }
                }
                
                if (j >= Ctx->N_y) { // Top boundary 
                    
                    // Adiabatic Boundary
                    if (Ctx->top_boundary == adiabatic) { 
                        irhs = 2*oned_end - j + 1;
                        u[k][j][i].comp[0] = -u[k][irhs][i].comp[0];
                        
                    }
                }
            }
        }
    }

    oned_begin = 0; oned_end = Ctx->N_z-1; // Finally in the z-direction
	
    for (k = zsg; k < zsg+zmg; ++k) {
        for (j = ysg; j < ysg+ymg; ++j) {
            for (i = xsg; i < xsg+xmg; ++i) {
                
                if (k < 0) { // Back boundary 
                    
                    // Adiabatic Boundary

                    if (Ctx->back_boundary == adiabatic) { 
                        irhs = oned_begin - 1 - k;
                        u[k][j][i].comp[0] = -u[irhs][j][i].comp[0]; 
                    }
                }
                
                if (k >= Ctx->N_z) { // Front boundary 
                    
                    // Adiabatic Boundary

                    if (Ctx->front_boundary == adiabatic) { 
                        irhs = 2*oned_end - k + 1;
                        u[k][j][i].comp[0] = -u[irhs][j][i].comp[0];
                
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------
    // Do WENO reconstruction for each cell
    //--------------------------------------------------------------

    for ( k = zs-1; k < zs+zm+1; ++k ) {
        for ( j = ys-1; j < ys+ym+1; ++j ) { 
            for ( i = xs-1; i < xs+xm+1; ++i ) {
                
                local_k = k - (zs-1);
                local_j = j - (ys-1); 
                local_i = i - (xs-1); 
                    
                // Select stencils for the solution

                    for (oned_begin = -2; oned_begin < 3; ++oned_begin) {
                        u_x_loc[oned_begin+2] = u[k][j][i+oned_begin].comp[0];
                        u_y_loc[oned_begin+2] = u[k][j+oned_begin][i].comp[0];
                        u_z_loc[oned_begin+2] = u[k+oned_begin][j][i].comp[0];
                    }
                    
                    u_xy_loc[0] = u[k][j][i].comp[0];
                    u_xy_loc[1] = u[k][j+1][i+1].comp[0];
                    u_xy_loc[2] = u[k][j-1][i+1].comp[0];
                    u_xy_loc[3] = u[k][j+1][i-1].comp[0];
                    u_xy_loc[4] = u[k][j-1][i-1].comp[0];
                    
                    u_yz_loc[0] = u[k][j][i].comp[0];
                    u_yz_loc[1] = u[k+1][j+1][i].comp[0];
                    u_yz_loc[2] = u[k-1][j+1][i].comp[0];
                    u_yz_loc[3] = u[k+1][j-1][i].comp[0];
                    u_yz_loc[4] = u[k-1][j-1][i].comp[0];
                    
                    u_zx_loc[0] = u[k][j][i].comp[0];
                    u_zx_loc[1] = u[k+1][j][i+1].comp[0];
                    u_zx_loc[2] = u[k+1][j][i-1].comp[0];
                    u_zx_loc[3] = u[k-1][j][i+1].comp[0];
                    u_zx_loc[4] = u[k-1][j][i-1].comp[0];
                    
                    u_xyz_loc[0] = u[k+1][j+1][i+1].comp[0]; u_xyz_loc[1] = u[k+1][j+1][i-1].comp[0];
                    u_xyz_loc[2] = u[k+1][j-1][i+1].comp[0]; u_xyz_loc[3] = u[k-1][j+1][i+1].comp[0];
                    u_xyz_loc[4] = u[k+1][j-1][i-1].comp[0]; u_xyz_loc[5] = u[k-1][j+1][i-1].comp[0];
                    u_xyz_loc[6] = u[k-1][j-1][i+1].comp[0]; u_xyz_loc[7] = u[k-1][j-1][i-1].comp[0];
                    
                    weno(u_x_loc, u_y_loc, u_z_loc, u_xy_loc, u_yz_loc, u_zx_loc, u_xyz_loc, u_coeffs);
                    
                    for (f = 0; f < N_gp2d; ++f) {
                        
                        evaluate_grad(u_coeffs, -0.5, x_gp2d[f], y_gp2d[f], Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,0,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,0,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,0,f,2,grad_z);
                        
                        evaluate_grad(u_coeffs, 0.5, x_gp2d[f], y_gp2d[f], Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,1,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,1,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,1,f,2,grad_z);
                        
                        evaluate_grad(u_coeffs, x_gp2d[f], -0.5, y_gp2d[f], Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,2,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,2,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,2,f,2,grad_z);

                        
                        evaluate_grad(u_coeffs, x_gp2d[f], 0.5, y_gp2d[f], Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,3,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,3,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,3,f,2,grad_z);
                    
                        evaluate_grad(u_coeffs, x_gp2d[f], y_gp2d[f], -0.5, Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,4,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,4,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,4,f,2,grad_z);
                        
                        evaluate_grad(u_coeffs, x_gp2d[f], y_gp2d[f], 0.5, Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,5,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,5,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,0,5,f,2,grad_z);
                    }
            }
        }
    } // End of cell loop

    //Find the upwind fluxes
    
    // in x-direction 
    
    nx = 1.0; ny = 0.0; nz = 0.0; 
    
    for (k=zs; k<zs+zm; ++k) {
        for (j=ys; j<ys+ym; ++j) {
            for (i=xs; i <xs+xm+1; ++i) {

                        
                local_k = k - (zs-1);
                local_j = j - (ys-1); 
                local_i = i - (xs-1); 
            
                Flux = 0.0;
 
                for (f = 0; f < N_gp2d; ++f) {
                        
                    for (iDim = 0; iDim < DIM; ++iDim) {
                    
                        grad_QL[0][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i-1, 0, 1, f, iDim);
                        grad_QR[0][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i,   0, 0, f, iDim);
                    }

                    RiemannSolver(grad_QL, grad_QR, nx, ny, nz, &F);

                    Flux += w_gp2d[f]*F;

                }

                set_element_3d(Ctx->F, k-zs, j-ys, i-xs, Flux); 
                
            }
        }
    }
    
    // in y-direction 
    
    nx = 0.0; ny = 1.0; nz = 0.0; 
    
    for (k=zs; k<zs+zm; ++k) {
        for (j=ys; j<ys+ym+1; ++j) {
            for (i=xs; i <xs+xm; ++i) {
                
                        
                local_k = k - (zs-1);
                local_j = j - (ys-1); 
                local_i = i - (xs-1); 
            
                Flux = 0.0;

                for (f = 0; f < N_gp2d; ++f) {
                
                    for (iDim = 0; iDim < DIM; ++iDim) {
                    
                        grad_QL[0][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j-1, local_i, 0, 3, f, iDim);
                        grad_QR[0][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i,   0, 2, f, iDim);
                    }

                    RiemannSolver(grad_QL, grad_QR, nx, ny, nz, &F);
                
                    Flux += w_gp2d[f]*F;
                    
                }
        
                set_element_3d(Ctx->G, k-zs, j-ys, i-xs, Flux); 
                
            }
        }
    }
    
    // in z-direction 
    
    nx = 0.0; ny = 0.0; nz = 1.0; 
    
    for (k=zs; k<zs+zm+1; ++k) {
        for (j=ys; j<ys+ym; ++j) {
            for (i=xs; i <xs+xm; ++i) {

                        
                local_k = k - (zs-1);
                local_j = j - (ys-1); 
                local_i = i - (xs-1); 
            
                Flux = 0.0; 
 
                for (f = 0; f < N_gp2d; ++f) {
                
                    for (iDim = 0; iDim < DIM; ++iDim) {
                    
                        grad_QL[0][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k-1, local_j, local_i, 0, 5, f, iDim);
                        grad_QR[0][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i,   0, 4, f, iDim);
                    }

                    RiemannSolver(grad_QL, grad_QR, nx, ny, nz, &F); 
                
                    Flux += w_gp2d[f]*F;
                }
        
                set_element_3d(Ctx->H, k-zs, j-ys, i-xs, Flux); 
            }
        }
    }
    
    // 4) Find the rhs in each cell  

    for(k=zs; k<zs+zm; ++k) {
        for (j=ys; j<ys+ym; ++j) {
            for (i=xs; i<xs+xm; ++i) {

                rhs[k][j][i].comp[0]  = u[k][j][i].comp[1];

                rhs[k][j][i].comp[1]  =     r1_h*(get_element_3d(Ctx->F, k-zs, j-ys, i+1-xs) - get_element_3d(Ctx->F, k-zs, j-ys, i-xs)) +
                                            r1_h*(get_element_3d(Ctx->G, k-zs, j+1-ys, i-xs) - get_element_3d(Ctx->G, k-zs, j-ys, i-xs)) + 
                                            r1_h*(get_element_3d(Ctx->H, k+1-zs, j-ys, i-xs) - get_element_3d(Ctx->H, k-zs, j-ys, i-xs));
                
            }
        }
    }


    ierr = DMDAVecRestoreArray(da,Ctx->localU,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,RHS,&rhs);CHKERRQ(ierr);

    return ierr; 
}
