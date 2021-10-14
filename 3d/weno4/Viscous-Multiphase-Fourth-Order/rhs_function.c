/*
 * rhs_function.c
 *      Author: sunder
 */ 
#include "hype.h" 


//----------------------------------------------------------------------------
// Find the cell averages of primitive variables in each cell 
//----------------------------------------------------------------------------

PetscErrorCode ComputePrimitiveVariables(Vec U, Vec W, DM da, AppCtx *Ctx) {
    
    PetscErrorCode ierr;                 // For catching PETSc errors 
    PetscInt c,i,j,k,xs,ys,zs,xm,ym,zm;  // Corners of the grid on the given solution 
    Field  ***u;                         // Local array of the conserved variables 
    Field  ***w;                         // Local array of the primitive variables 

    Field Q; Field V;  

    // Read the local solution to the array u  

    ierr = DMDAVecGetArrayRead(da, U, &u);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, W, &w);CHKERRQ(ierr);

    ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);

    for (k=zs; k<zs+zm; ++k) {
        for (j=ys; j<ys+ym; ++j) {
            for (i=xs; i<xs+xm; ++i) {
            
                // Select stencils for each component of the solution 
        
                for (c = 0 ; c < nVar; ++c)
                    Q.comp[c] = u[k][j][i].comp[c];
        
                PDECons2Prim(&Q, &V);
                
                for (c = 0 ; c < nVar; ++c)
                    w[k][j][i].comp[c] = V.comp[c];
            }
        }  
    }

    ierr = DMDAVecRestoreArrayRead(da, U, &u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,W,&w);CHKERRQ(ierr);

    return ierr; 
}

//----------------------------------------------------------------------------
// Compute the value of RHS for each cell in the domain using 
// conserved variables
//----------------------------------------------------------------------------

PetscErrorCode RHSFunctionPrimitive(TS ts, PetscReal t, Vec U, Vec RHS, void* ctx) {
    
    PetscErrorCode ierr;           
    AppCtx *Ctx = (AppCtx*)ctx; 
    DM da;                         
    PetscInt c,i,j,k,l,m,f,xs,ys,zs,xm,ym,zm,xsg,ysg,zsg,xmg,ymg,zmg,oned_begin,oned_end,irhs,iDim;         
    
    Vec W;
    Field   ***w;                   
    Field   ***rhs;
    PetscReal grad_x, grad_y, grad_z;
    PetscReal s_c, s_max_c = 0.0;
    PetscReal s_v, s_max_v = 0.0; 
    PetscReal w_x_loc[5], w_y_loc[5], w_z_loc[5], w_xy_loc[5], w_yz_loc[5], w_zx_loc[5], w_xyz_loc[8];
    PetscReal dt, temp; 
    PetscReal sol[dofs_per_cell][nVar], u_sol[dofs_per_cell][nVar], coeffs[dofs_per_cell], u_coeffs[dofs_per_cell];
    PetscReal r1_h = 1./(Ctx->h); 
    PetscReal nx, ny, nz, xloc, yloc, zloc; 
    PetscInt local_i, local_j, local_k;
    PetscBool PAD; 
    Field V_node; 
    Field VL; Field VR;
    PetscReal grad_VL[nVar][DIM]; PetscReal grad_VR[nVar][DIM]; 
    PetscReal Flux[7], Flux_conv[7];
    Field Flux_visc;
    
    ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
    
    // Scatter global->local to have access to the required ghost values 
    
    ierr = DMCreateGlobalVector(da, &W);CHKERRQ(ierr);
    ComputePrimitiveVariables(U, W, da, Ctx);

    //ierr = DMGlobalToLocalBegin(da, U, INSERT_VALUES, Ctx->localU);CHKERRQ(ierr);
    //ierr = DMGlobalToLocalEnd(da, U, INSERT_VALUES,   Ctx->localU);CHKERRQ(ierr);
    
    ierr = DMGlobalToLocalBegin(da, W, INSERT_VALUES, Ctx->localU); CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(da, W, INSERT_VALUES,   Ctx->localU); CHKERRQ(ierr);

    // Read the local solution to the array u  

    ierr = DMDAVecGetArrayRead(da, Ctx->localU, &w); CHKERRQ(ierr); 
    ierr = DMDAVecGetArray(da, RHS, &rhs);CHKERRQ(ierr);

    ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
    ierr = DMDAGetGhostCorners(da, &xsg, &ysg, &zsg, &xmg, &ymg, &zmg);
    
    // 1) Apply boundary conditions (Periodic boundary conditions are taken care of by PETsc)

    oned_begin = 0; oned_end = Ctx->N_x-1;  // First in the x-direction 

    for (k = zsg; k < zsg+zmg; ++k) {
        for (j = ysg; j < ysg+ymg; ++j) {
            for (i = xsg; i < xsg+xmg; ++i) {
                
                if (i < 0)  { // Left boundary 
                    
                    if (Ctx->left_boundary == transmissive) { // Transmissive/Outflow Boundary
                        irhs = oned_begin; 
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[k][j][irhs].comp[c];
                    }
                    
                    if (Ctx->left_boundary == reflective) { // Reflective Boundary 
                        irhs = oned_begin - 1 - i; 
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[k][j][irhs].comp[c];
                        
                        w[k][j][i].comp[1] = -w[k][j][i].comp[1]; // Reflect x-momentum component 
                    }
                }
                
                if (i >= Ctx->N_x) { // Right Boundary 
                    
                    if (Ctx->right_boundary == transmissive) { // Transmissive/Outflow Boundary 
                        irhs = oned_end; 
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[k][j][irhs].comp[c];
                    }
                    
                    if (Ctx->right_boundary == reflective) { // Reflective Boundary
                        irhs = 2*oned_end - i + 1; 
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[k][j][irhs].comp[c];
                        
                        w[k][j][i].comp[1] = -w[k][j][i].comp[1]; // Reflect x-momentum component 
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
                    
                    if (Ctx->bottom_boundary == transmissive) { // Transmissive/Outflow Boundary
                        irhs = oned_begin;
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[k][irhs][i].comp[c]; 
                    }
                    
                    if (Ctx->bottom_boundary == reflective) { // Reflective Boundary
                        irhs = oned_begin - 1 - j;
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[k][irhs][i].comp[c]; 
                        
                        w[k][j][i].comp[2] = -w[k][j][i].comp[2]; // Reflect y-momentum component
                    }
                }
                
                if (j >= Ctx->N_y) { // Top boundary 
                    
                    if (Ctx->top_boundary == transmissive) { // Transmissive/Outflow Boundary
                        irhs = oned_end;
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[k][irhs][i].comp[c];
                    }
                    
                    if (Ctx->top_boundary == reflective) { // Reflective Boundary
                        irhs = 2*oned_end - j + 1;
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[k][irhs][i].comp[c];
                        
                        w[k][j][i].comp[2] = -w[k][j][i].comp[2]; // Reflect y-momentum component
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
                    
                    if (Ctx->back_boundary == transmissive) { // Transmissive/Outflow Boundary
                        irhs = oned_begin;
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[irhs][j][i].comp[c]; 
                    }
                    
                    if (Ctx->back_boundary == reflective) { // Reflective Boundary
                        irhs = oned_begin - 1 - k;
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[irhs][j][i].comp[c]; 
                        
                        w[k][j][i].comp[3] = -w[k][j][i].comp[3]; // Reflect z-momentum component
                    }
                }
                
                if (k >= Ctx->N_z) { // Front boundary 
                    
                    if (Ctx->front_boundary == transmissive) { // Transmissive/Outflow Boundary
                        irhs = oned_end;
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[irhs][j][i].comp[c];
                    }
                    
                    if (Ctx->front_boundary == reflective) { // Reflective Boundary
                        irhs = 2*oned_end - k + 1;
                        for (c = 0; c < nVar; ++c)
                            w[k][j][i].comp[c] = w[irhs][j][i].comp[c];
                        
                        w[k][j][i].comp[3] = -w[k][j][i].comp[3]; // Reflect z-momentum component
                    }
                }
            }
        }
    }
    
    // 2) Find the boundary extrapolated values of primitive variables in all the cells 
    
    for ( k = zs-1; k < zs+zm+1; ++k ) {
        for ( j = ys-1; j < ys+ym+1; ++j ) { 
            for ( i = xs-1; i < xs+xm+1; ++i ) {
                
                local_k = k - (zs-1);
                local_j = j - (ys-1); 
                local_i = i - (xs-1);
                PAD = PETSC_TRUE; 
                
                for (c = 0; c < nVar; ++c) {
                    
                    for (oned_begin = -2; oned_begin < 3; ++oned_begin) {
                        w_x_loc[oned_begin+2] = w[k][j][i+oned_begin].comp[c];
                        w_y_loc[oned_begin+2] = w[k][j+oned_begin][i].comp[c];
                        w_z_loc[oned_begin+2] = w[k+oned_begin][j][i].comp[c];
                    }
                    
                    w_xy_loc[0] = w[k][j][i].comp[c];
                    w_xy_loc[1] = w[k][j+1][i+1].comp[c];
                    w_xy_loc[2] = w[k][j-1][i+1].comp[c];
                    w_xy_loc[3] = w[k][j+1][i-1].comp[c];
                    w_xy_loc[4] = w[k][j-1][i-1].comp[c];
                    
                    w_yz_loc[0] = w[k][j][i].comp[c];
                    w_yz_loc[1] = w[k+1][j+1][i].comp[c];
                    w_yz_loc[2] = w[k-1][j+1][i].comp[c];
                    w_yz_loc[3] = w[k+1][j-1][i].comp[c];
                    w_yz_loc[4] = w[k-1][j-1][i].comp[c];
                    
                    w_zx_loc[0] = w[k][j][i].comp[c];
                    w_zx_loc[1] = w[k+1][j][i+1].comp[c];
                    w_zx_loc[2] = w[k+1][j][i-1].comp[c];
                    w_zx_loc[3] = w[k-1][j][i+1].comp[c];
                    w_zx_loc[4] = w[k-1][j][i-1].comp[c];
                    
                    w_xyz_loc[0] = w[k+1][j+1][i+1].comp[c]; w_xyz_loc[1] = w[k+1][j+1][i-1].comp[c];
                    w_xyz_loc[2] = w[k+1][j-1][i+1].comp[c]; w_xyz_loc[3] = w[k-1][j+1][i+1].comp[c];
                    w_xyz_loc[4] = w[k+1][j-1][i-1].comp[c]; w_xyz_loc[5] = w[k-1][j+1][i-1].comp[c];
                    w_xyz_loc[6] = w[k-1][j-1][i+1].comp[c]; w_xyz_loc[7] = w[k-1][j-1][i-1].comp[c];
                    
                    weno(w_x_loc, w_y_loc, w_z_loc, w_xy_loc, w_yz_loc, w_zx_loc, w_xyz_loc, u_coeffs, coeffs);
                    
                    // Store coefficients 
                    
                    for (l = 0; l < dofs_per_cell; ++l) {
                        sol[l][c] = coeffs[l];
                        u_sol[l][c] = u_coeffs[l];
                    }
                } // Reconstruction loop  
                
                // Check Pressure/Density positivity at each node 
                
                for (m = 0; m < N_Node; ++m) {
                    
                    for (c = 0; c < nVar; ++c) {
                        for (l = 0; l < dofs_per_cell; ++l) {
                            coeffs[l] = sol[l][c]; 
                        }
                        V_node.comp[c] = evaluate_polynomial(xNode[m], yNode[m], zNode[m], coeffs);
                    }
                    
                    PAD = PDECheckPAD(V_node);
                    
                    if (PAD == PETSC_FALSE)
                        break; 
                }
                
                // If physical admissibility is violated, reduce to first order 
                
                if (PAD == PETSC_FALSE) {
                    for (c = 0; c < nVar; ++c) {
                        for (l = 1; l < dofs_per_cell; ++l) {
                            sol[l][c] = 0.0; 
                        }
                    }
                }
                
                // Find boundary extrapolated values 
                
                for (c = 0; c < nVar; ++c) {
                        
                    for (l = 0; l < dofs_per_cell; ++l) {
                        coeffs[l] = sol[l][c];
                        u_coeffs[l] = u_sol[l][c];
                    }
                    
                    for (f = 0; f < N_gp2d; ++f) {
                        
                        temp = evaluate_polynomial(-0.5, x_gp2d[f], y_gp2d[f], coeffs); // Left face
                        evaluate_grad(u_coeffs, -0.5, x_gp2d[f], y_gp2d[f], Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_6d(Ctx->u_bnd,local_k,local_j,local_i,c,0,f,temp);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,0,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,0,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,0,f,2,grad_z);
                        
                        temp = evaluate_polynomial( 0.5, x_gp2d[f], y_gp2d[f], coeffs); // Right face
                        evaluate_grad(u_coeffs, 0.5, x_gp2d[f], y_gp2d[f], Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_6d(Ctx->u_bnd,local_k,local_j,local_i,c,1,f,temp);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,1,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,1,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,1,f,2,grad_z);
                        
                        temp = evaluate_polynomial(x_gp2d[f], -0.5, y_gp2d[f], coeffs); // Bottom face
                        evaluate_grad(u_coeffs, x_gp2d[f], -0.5, y_gp2d[f], Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_6d(Ctx->u_bnd,local_k,local_j,local_i,c,2,f,temp);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,2,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,2,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,2,f,2,grad_z);

                        temp = evaluate_polynomial(x_gp2d[f], 0.5, y_gp2d[f], coeffs);  // Top face
                        evaluate_grad(u_coeffs, x_gp2d[f], 0.5, y_gp2d[f], Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_6d(Ctx->u_bnd,local_k,local_j,local_i,c,3,f,temp);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,3,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,3,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,3,f,2,grad_z);
                    
                        temp = evaluate_polynomial(x_gp2d[f], y_gp2d[f], -0.5, coeffs); // Back face
                        evaluate_grad(u_coeffs, x_gp2d[f], y_gp2d[f], -0.5, Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_6d(Ctx->u_bnd,local_k,local_j,local_i,c,4,f,temp);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,4,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,4,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,4,f,2,grad_z);
                        
                        temp = evaluate_polynomial(x_gp2d[f], y_gp2d[f], 0.5, coeffs);  // Front face
                        evaluate_grad(u_coeffs, x_gp2d[f], y_gp2d[f], 0.5, Ctx->h, &grad_x, &grad_y, &grad_z);
                        set_element_6d(Ctx->u_bnd,local_k,local_j,local_i,c,5,f,temp);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,5,f,0,grad_x);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,5,f,1,grad_y);
                        set_element_7d(Ctx->u_bnd_grad,local_k,local_j,local_i,c,5,f,2,grad_z);
                    
                    } // Face loop
                    
                } // Variable loop 
                
            } // i loop
        } // j loop 
    } // k loop 

    // 3) Find the upwind fluxes 
    
    // in x-direction 
    
    nx = 1.0; ny = 0.0; nz = 0.0; 
    
    for (k=zs; k<zs+zm; ++k) {
        for (j=ys; j<ys+ym; ++j) {
            for (i=xs; i <xs+xm+1; ++i) {
                
                xloc = Ctx->x_min+(PetscReal)i*(Ctx->h); yloc = Ctx->y_min+(PetscReal)j*(Ctx->h); zloc = Ctx->z_min+(PetscReal)k*(Ctx->h);
                        
                local_k = k - (zs-1);
                local_j = j - (ys-1); 
                local_i = i - (xs-1); 
            
                for (c = 0; c < 7; ++c)
                    Flux[c] = 0.0; 
 
                for (f = 0; f < N_gp2d; ++f) {
                
                    for (c = 0; c < nVar; ++c) {
                        
                        VL.comp[c] = get_element_6d(Ctx->u_bnd, local_k, local_j, local_i-1, c, 1, f);
                        VR.comp[c] = get_element_6d(Ctx->u_bnd, local_k, local_j, local_i,   c, 0, f);
                        
                        for (iDim = 0; iDim < DIM; ++iDim) {
                        
                            grad_VL[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i-1, c, 1, f, iDim);
                            grad_VR[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i,   c, 0, f, iDim);
                        }
                    }
                
                    s_c = PDErotHLLCRiemannSolver(&VL, &VR, nx, ny, nz, xloc, yloc, zloc, Flux_conv);   if (s_c>s_max_c) s_max_c = s_c;
                    s_v = PDEViscRiemannSolverPrim(&VL, grad_VL, &VR, grad_VR, nx, ny, nz, &Flux_visc); if (s_v>s_max_v) s_max_v = s_v; 
                 
                
                    for (c = 0; c < 7; ++c) {
                        Flux[c] += w_gp2d[f]*(Flux_conv[c]);
                        if (c!=6)
                            Flux[c] += w_gp2[f]*Flux_visc.comp[c];
                    }
                }
        
                for (c = 0; c < 7; ++c) {
                    set_element_4d(Ctx->F, k-zs, j-ys, i-xs, c, Flux[c]); 
                }
            }
        }
    }
    
    // in y-direction 
    
    nx = 0.0; ny = 1.0; nz = 0.0; 
    
    for (k=zs; k<zs+zm; ++k) {
        for (j=ys; j<ys+ym+1; ++j) {
            for (i=xs; i <xs+xm; ++i) {
                
                xloc = Ctx->x_min+(PetscReal)i*(Ctx->h); yloc = Ctx->y_min+(PetscReal)j*(Ctx->h); zloc = Ctx->z_min+(PetscReal)k*(Ctx->h);
                        
                local_k = k - (zs-1);
                local_j = j - (ys-1); 
                local_i = i - (xs-1); 
            
                for (c = 0; c < 7; ++c)
                    Flux[c] = 0.0; 
 
                for (f = 0; f < N_gp2d; ++f) {
                
                    for (c = 0; c < nVar; ++c) {
                        
                        VL.comp[c] = get_element_6d(Ctx->u_bnd, local_k, local_j-1, local_i, c, 3, f);
                        VR.comp[c] = get_element_6d(Ctx->u_bnd, local_k, local_j, local_i,   c, 2, f);
                        
                        for (iDim = 0; iDim < DIM; ++iDim) {
                        
                            grad_VL[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j-1, local_i, c, 3, f, iDim);
                            grad_VR[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i,   c, 2, f, iDim);
                        }
                    }
                
                    s_c = PDErotHLLCRiemannSolver(&VL, &VR, nx, ny, nz, xloc, yloc, zloc, Flux_conv);   if (s_c>s_max_c) s_max_c = s_c;
                    s_v = PDEViscRiemannSolverPrim(&VL, grad_VL, &VR, grad_VR, nx, ny, nz, &Flux_visc); if (s_v>s_max_v) s_max_v = s_v;
                 
                
                    for (c = 0; c < 7; ++c) {
                        Flux[c] += w_gp2d[f]*(Flux_conv[c]);
                        if (c!=6)
                            Flux[c] += w_gp2[f]*Flux_visc.comp[c];
                    }
                }
        
                for (c = 0; c < 7; ++c) {
                    set_element_4d(Ctx->G, k-zs, j-ys, i-xs, c, Flux[c]); 
                }
            }
        }
    }
    
    // in z-direction 
    
    nx = 0.0; ny = 0.0; nz = 1.0; 
    
    for (k=zs; k<zs+zm+1; ++k) {
        for (j=ys; j<ys+ym; ++j) {
            for (i=xs; i <xs+xm; ++i) {
                
                xloc = Ctx->x_min+(PetscReal)i*(Ctx->h); yloc = Ctx->y_min+(PetscReal)j*(Ctx->h); zloc = Ctx->z_min+(PetscReal)k*(Ctx->h);
                        
                local_k = k - (zs-1);
                local_j = j - (ys-1); 
                local_i = i - (xs-1); 
            
                for (c = 0; c < 7; ++c)
                    Flux[c] = 0.0; 
 
                for (f = 0; f < N_gp2d; ++f) {
                
                    for (c = 0; c < nVar; ++c) {
                        
                        VL.comp[c] = get_element_6d(Ctx->u_bnd, local_k-1, local_j, local_i, c, 5, f);
                        VR.comp[c] = get_element_6d(Ctx->u_bnd, local_k, local_j, local_i,   c, 4, f);
                        
                        for (iDim = 0; iDim < DIM; ++iDim) {
                        
                            grad_VL[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k-1, local_j, local_i, c, 5, f, iDim);
                            grad_VR[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i,   c, 4, f, iDim);
                        }
                    }
                
                    s_c = PDErotHLLCRiemannSolver(&VL, &VR, nx, ny, nz, xloc, yloc, zloc, Flux_conv);   if (s_c>s_max_c) s_max_c = s_c; 
                    s_v = PDEViscRiemannSolverPrim(&VL, grad_VL, &VR, grad_VR, nx, ny, nz, &Flux_visc); if (s_v>s_max_v) s_max_v = s_v;
                 
                
                    for (c = 0; c < 7; ++c) {
                        Flux[c] += w_gp2d[f]*(Flux_conv[c]);
                        if (c!=6)
                            Flux[c] += w_gp2[f]*Flux_visc.comp[c];
                    }
                }
        
                for (c = 0; c < 7; ++c) {
                    set_element_4d(Ctx->H, k-zs, j-ys, i-xs, c, Flux[c]); 
                }
            }
        }
    }
    
    // 4) Find the rhs in each cell 

    for(k=zs; k<zs+zm; ++k) {
        for (j=ys; j<ys+ym; ++j) {
            for (i=xs; i<xs+xm; ++i) {

                for (c = 0 ; c < nVar; ++c) {
                    
                    if (c == 5) {
                    
                        rhs[k][j][i].comp[c]  = -r1_h*((get_element_4d(Ctx->F, k-zs, j-ys, i+1-xs, c) - 
                                                        w[k][j][i].comp[c]*get_element_4d(Ctx->F, k-zs, j-ys, i+1-xs, c+1))
                                                      -(get_element_4d(Ctx->F, k-zs, j-ys, i-xs, c) - 
                                                        w[k][j][i].comp[c]*get_element_4d(Ctx->F, k-zs, j-ys, i-xs, c+1)) )
                        
                                                -r1_h*((get_element_4d(Ctx->G, k-zs, j+1-ys, i-xs, c) - 
                                                        w[k][j][i].comp[c]*get_element_4d(Ctx->G, k-zs, j+1-ys, i-xs, c+1))
                                                      -(get_element_4d(Ctx->G, k-zs, j-ys, i-xs, c) - 
                                                        w[k][j][i].comp[c]*get_element_4d(Ctx->G, k-zs, j-ys, i-xs, c+1)) )
                                                
                                                -r1_h*((get_element_4d(Ctx->H, k+1-zs, j-ys, i-xs, c) - 
                                                        w[k][j][i].comp[c]*get_element_4d(Ctx->H, k+1-zs, j-ys, i-xs, c+1))
                                                      -(get_element_4d(Ctx->H, k-zs, j-ys, i-xs, c) - 
                                                        w[k][j][i].comp[c]*get_element_4d(Ctx->H, k-zs, j-ys, i-xs, c+1)) );
                        
                    }
                    
                    else {
        
                        rhs[k][j][i].comp[c]  = -r1_h*(get_element_4d(Ctx->F, k-zs, j-ys, i+1-xs, c) - get_element_4d(Ctx->F, k-zs, j-ys, i-xs, c))
                                                -r1_h*(get_element_4d(Ctx->G, k-zs, j+1-ys, i-xs, c) - get_element_4d(Ctx->G, k-zs, j-ys, i-xs, c))
                                                -r1_h*(get_element_4d(Ctx->H, k+1-zs, j-ys, i-xs, c) - get_element_4d(Ctx->H, k-zs, j-ys, i-xs, c));
                    }
                        
                }
            }
        }
    }
    
    ierr = DMDAVecRestoreArray(da,Ctx->localU,&w);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,RHS,&rhs);CHKERRQ(ierr);

    dt = (1.0/3.0)*(Ctx->CFL*Ctx->h)/(s_max_c + (r1_h*s_max_v)*2.0); // 1/3 is for three dimensions

    ierr = MPI_Allreduce(&dt, &Ctx->dt, 1, MPIU_REAL,MPIU_MIN, PetscObjectComm((PetscObject)da)); CHKERRQ(ierr);

    return ierr; 
}
