/*
 * rhs_function.c
 *      Author: sunder
 */ 
#include "hype.h" 

//----------------------------------------------------------------------------
// Compute the value of RHS for each cell in the domain using 
// conserved variables
//----------------------------------------------------------------------------

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec RHS, void* ctx) {

    PetscErrorCode ierr;           
    AppCtx *Ctx = (AppCtx*)ctx; 
    DM da;                         
    PetscInt i,j,k,f,q,iDim,xs,ys,xm,ym,xs_g,ys_g,xm_g,ym_g,oned_begin,oned_end,irhs;                
    Field   **u;                   
    Field   **rhs;                 
    PetscReal u_x_loc[s_width], u_y_loc[s_width], u_xy_loc[s_width];  
    PetscReal coeffs[nDOF];
    PetscReal grad_x, grad_y; 
    PetscReal r1_h = 1./(Ctx->h);
    PetscReal r1_h2 = 1./((Ctx->h)*(Ctx->h));  
    PetscReal nx, ny; 
    PetscInt local_i, local_j;

    PetscReal grad_QL[nVar][DIM]; PetscReal grad_QR[nVar][DIM];
    PetscReal Flux, F; 

    ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
    
    // Scatter global->local to have access to the required ghost values 

    ierr = DMGlobalToLocalBegin(da, U, INSERT_VALUES, Ctx->localU);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, U, INSERT_VALUES,   Ctx->localU);CHKERRQ(ierr);

    // Read the local solution to the array u  

    ierr = DMDAVecGetArrayRead(da, Ctx->localU, &u); CHKERRQ(ierr); 
    ierr = DMDAVecGetArray(da, RHS,&rhs);CHKERRQ(ierr);

    ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
    ierr = DMDAGetGhostCorners(da, &xs_g, &ys_g, NULL, &xm_g, &ym_g, NULL);

    //--------------------------------------------------------------
    // Apply Boundary Conditions 
    //--------------------------------------------------------------

    oned_begin = 0; oned_end = Ctx->N_x-1; 

    for (j = ys_g; j < ys_g+ym_g; ++j) {
        for (i = xs_g; i < xs_g+xm_g; ++i) {
            
            if (i < 0)  { // Left boundary 
            
                
                // Adiabatic Wall Boundary 
                
                if (Ctx->left_boundary == adiabatic_wall) {
                    
                    irhs = oned_begin - 1 - i; 
                
                    u[j][i].comp[0] = u[j][irhs].comp[0];
                    u[j][i].comp[0] = -u[j][i].comp[0];
                }
                
            }
            
            if (i >= Ctx->N_x) { // Right Boundary 
                
                // Adiabatic Wall Boundary
                
                if (Ctx->right_boundary == adiabatic_wall) {
                    
                    irhs = 2*oned_end - i + 1; 
                    
                    u[j][i].comp[0] = u[j][irhs].comp[0];
                    u[j][i].comp[0] = -u[j][i].comp[0];
                    
                }
            }
        }
    }

    oned_begin = 0; oned_end = Ctx->N_y-1;

    for (j = ys_g; j < ys_g+ym_g; ++j) {
        for (i = xs_g; i < xs_g+xm_g; ++i) {
            
            if (j < 0) { // Bottom boundary
                
                // Adiabatic Wall Boundary
                
                if (Ctx->bottom_boundary == adiabatic_wall) {
                
                    irhs = oned_begin - 1 - j;
                    
                    u[j][i].comp[0] = u[irhs][i].comp[0]; 
                    u[j][i].comp[0] = -u[j][i].comp[0];
                    
                }
            }
            
            if (j >= Ctx->N_y) { // Top boundary 
                
                // Adiabatic Wall Boundary
                
                if (Ctx->top_boundary == adiabatic_wall) {
                
                    irhs = 2*oned_end - j + 1;;
                    
                    u[j][i].comp[0] = u[irhs][i].comp[0];
                    u[j][i].comp[0] = -u[j][i].comp[0];
                }
            }
        }
    }

    //--------------------------------------------------------------
    // Do WENO reconstruction for each cell
    //--------------------------------------------------------------
    
    for (j=ys-1; j<ys+ym+1; j++) {
        for (i=xs-1; i<xs+xm+1; i++) {
            
            local_j = j - (ys-1); 
            local_i = i - (xs-1);
            
            // Select stencils for each component of the solution 
        
            u_x_loc[0] = u[j][i-2].comp[0]; 
            u_x_loc[1] = u[j][i-1].comp[0]; 
            u_x_loc[2] = u[j][i].comp[0]; 
            u_x_loc[3] = u[j][i+1].comp[0]; 
            u_x_loc[4] = u[j][i+2].comp[0];
            
            u_y_loc[0] = u[j-2][i].comp[0]; 
            u_y_loc[1] = u[j-1][i].comp[0]; 
            u_y_loc[2] = u[j][i].comp[0]; 
            u_y_loc[3] = u[j+1][i].comp[0]; 
            u_y_loc[4] = u[j+2][i].comp[0];

            u_xy_loc[0] = u[j][i].comp[0];
            u_xy_loc[1] = u[j+1][i+1].comp[0];
            u_xy_loc[2] = u[j-1][i+1].comp[0];
            u_xy_loc[3] = u[j+1][i-1].comp[0];
            u_xy_loc[4] = u[j-1][i-1].comp[0];
            
            weno(u_x_loc, u_y_loc, u_xy_loc, coeffs);
                
            // Calculate boundary extrpolated gradients 
                
            for (f = 0; f < 4; ++f) {
                
                for (q = 0; q < N_gp2; ++q) {
                    
                    grad_x = 0.0; grad_y = 0.0; 
                
                    for (k = 0; k < nDOF; ++k) {
                        grad_x += coeffs[k]*get_element_3d(Ctx->gradphiFace_x,f,q,k);
                        grad_y += coeffs[k]*get_element_3d(Ctx->gradphiFace_y,f,q,k);
                    }
                    
                    grad_x = r1_h*grad_x; grad_y = r1_h*grad_y;
                    
                    set_element_5d(Ctx->u_bnd_grad, local_j, local_i, f, q, 0, grad_x);
                    set_element_5d(Ctx->u_bnd_grad, local_j, local_i, f, q, 1, grad_y);
                }
            }
                
        }
    } // End of cell loop
    
    // Find the upwind flux on each face in x-dirction 
    
    nx = 1.0; ny = 0.0; 

    for (j = ys; j < ys+ym; ++j) {
        for (i = xs; i < xs+xm+1; ++i) {
            
            local_j = j - (ys-1); 
            local_i = i - (xs-1);
            
            Flux = 0.0; 

            for (q = 0; q < N_gp2; ++q) {
                
                for (iDim = 0; iDim < DIM; ++iDim) {
                    grad_QL[0][iDim] = get_element_5d(Ctx->u_bnd_grad, local_j, local_i-1, 1, q, iDim);
                    grad_QR[0][iDim] = get_element_5d(Ctx->u_bnd_grad, local_j, local_i,   0, q, iDim);
                }
                
                LLFRiemannSolver(grad_QL, grad_QR, nx, ny, &F);
                
                Flux += w_gp2[q]*F;
            
            }
        
            set_element_2d(Ctx->F, j-ys, i-xs, Flux);
        }
    }
            
    // Find the upwind flux on each face in y-dirction 
    
    nx = 0.0; ny = 1.0;

    for (j = ys; j < ys+ym+1; ++j) {
        for (i = xs; i < xs+xm; ++i) {
            
            local_j = j - (ys-1); 
            local_i = i - (xs-1); 

            Flux = 0.0; 
            
            for (q = 0; q < N_gp2; ++q) {
                
                for (iDim = 0; iDim < DIM; ++iDim) {
                    grad_QL[0][iDim] = get_element_5d(Ctx->u_bnd_grad, local_j-1, local_i, 3, q, iDim);
                    grad_QR[0][iDim] = get_element_5d(Ctx->u_bnd_grad, local_j,   local_i, 2, q, iDim);
                }

                LLFRiemannSolver(grad_QL, grad_QR, nx, ny, &F);
                
                Flux += w_gp2[q]*F;
            }
            
            set_element_2d(Ctx->G, j-ys, i-xs, Flux);
            
        }
    }

    // Now find the rhs in each cell 

    for (j=ys; j<ys+ym; ++j) {
        for (i=xs; i<xs+xm; ++i) {

        
            rhs[j][i].comp[0] = u[j][i].comp[1]; 
            
            rhs[j][i].comp[1] = r1_h*(get_element_2d(Ctx->F, j-ys, i+1-xs) - get_element_2d(Ctx->F, j-ys, i-xs)) + 
                                r1_h*(get_element_2d(Ctx->G, j+1-ys, i-xs) - get_element_2d(Ctx->G, j-ys, i-xs)) ;
            
            /*add source at (isx, isy)*/
            if (i == Ctx->isx && j == Ctx->isy ) 
                rhs[j][i].comp[1] += Source(t)*r1_h2;
                   
        }
    }

    ierr = DMDAVecRestoreArray(da,Ctx->localU,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,RHS,&rhs);CHKERRQ(ierr);

    return ierr; 
}

