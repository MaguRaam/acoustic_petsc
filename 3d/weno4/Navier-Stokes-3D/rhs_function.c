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
    PetscInt c,i,j,k,f,xs,ys,zs,xm,ym,zm,xsg,ysg,zsg,xmg,ymg,zmg,oned_begin, oned_end,irhs,iDim;         
    
    Field   ***u;                   
    Field   ***rhs;                 
    PetscReal grad_x, grad_y, grad_z;
    PetscReal s_c, s_max_c = 0.0;
    PetscReal s_v, s_max_v = 0.0; 
    PetscReal u_x_loc[5], u_y_loc[5], u_z_loc[5], u_xy_loc[5], u_yz_loc[5], u_zx_loc[5], u_xyz_loc[8];
    PetscReal dt, temp; 
    PetscReal coeffs[dofs_per_cell];
    PetscReal u_coeffs[dofs_per_cell];
    PetscReal r1_h = 1./(Ctx->h); 
    PetscReal nx, ny, nz, xloc, yloc, zloc; 
    PetscInt local_i, local_j, local_k;
    
    Field QL; Field QR;
    PetscReal grad_QL[nVar][DIM]; PetscReal grad_QR[nVar][DIM];
    Field Flux, Flux_conv; 
    Field Flux_visc;
    
    ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
    
    // Scatter global->local to have access to the required ghost values 

    ierr = DMGlobalToLocalBegin(da, U, INSERT_VALUES, Ctx->localU);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, U, INSERT_VALUES,   Ctx->localU);CHKERRQ(ierr);

    // Read the local solution to the array u  

    ierr = DMDAVecGetArrayRead(da, Ctx->localU, &u); CHKERRQ(ierr); 
    ierr = DMDAVecGetArray(da, RHS,&rhs);CHKERRQ(ierr);

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
                            u[k][j][i].comp[c] = u[k][j][irhs].comp[c];
                    }
                    
                    if (Ctx->left_boundary == reflective) { // Reflective Boundary 
                        irhs = oned_begin - 1 - i; 
                        for (c = 0; c < nVar; ++c)
                            u[k][j][i].comp[c] = u[k][j][irhs].comp[c];
                        
                        u[k][j][i].comp[1] = -u[k][j][i].comp[1]; // Reflect x-momentum component 
                    }
                }
                
                if (i >= Ctx->N_x) { // Right Boundary 
                    
                    if (Ctx->right_boundary == transmissive) { // Transmissive/Outflow Boundary 
                        irhs = oned_end; 
                        for (c = 0; c < nVar; ++c)
                            u[k][j][i].comp[c] = u[k][j][irhs].comp[c];
                    }
                    
                    if (Ctx->right_boundary == reflective) { // Reflective Boundary
                        irhs = 2*oned_end - i + 1; 
                        for (c = 0; c < nVar; ++c)
                            u[k][j][i].comp[c] = u[k][j][irhs].comp[c];
                        
                        u[k][j][i].comp[1] = -u[k][j][i].comp[1]; // Reflect x-momentum component 
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
                            u[k][j][i].comp[c] = u[k][irhs][i].comp[c]; 
                    }
                    
                    if (Ctx->bottom_boundary == reflective) { // Reflective Boundary
                        irhs = oned_begin - 1 - j;
                        for (c = 0; c < nVar; ++c)
                            u[k][j][i].comp[c] = u[k][irhs][i].comp[c]; 
                        
                        u[k][j][i].comp[2] = -u[k][j][i].comp[2]; // Reflect y-momentum component
                    }
                }
                
                if (j >= Ctx->N_y) { // Top boundary 
                    
                    if (Ctx->top_boundary == transmissive) { // Transmissive/Outflow Boundary
                        irhs = oned_end;
                        for (c = 0; c < nVar; ++c)
                            u[k][j][i].comp[c] = u[k][irhs][i].comp[c];
                    }
                    
                    if (Ctx->top_boundary == reflective) { // Reflective Boundary
                        irhs = 2*oned_end - j + 1;
                        for (c = 0; c < nVar; ++c)
                            u[k][j][i].comp[c] = u[k][irhs][i].comp[c];
                        
                        u[k][j][i].comp[2] = -u[k][j][i].comp[2]; // Reflect y-momentum component
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
                            u[k][j][i].comp[c] = u[irhs][j][i].comp[c]; 
                    }
                    
                    if (Ctx->back_boundary == reflective) { // Reflective Boundary
                        irhs = oned_begin - 1 - k;
                        for (c = 0; c < nVar; ++c)
                            u[k][j][i].comp[c] = u[irhs][j][i].comp[c]; 
                        
                        u[k][j][i].comp[3] = -u[k][j][i].comp[3]; // Reflect z-momentum component
                    }
                }
                
                if (k >= Ctx->N_z) { // Front boundary 
                    
                    if (Ctx->front_boundary == transmissive) { // Transmissive/Outflow Boundary
                        irhs = oned_end;
                        for (c = 0; c < nVar; ++c)
                            u[k][j][i].comp[c] = u[irhs][j][i].comp[c];
                    }
                    
                    if (Ctx->front_boundary == reflective) { // Reflective Boundary
                        irhs = 2*oned_end - k + 1;
                        for (c = 0; c < nVar; ++c)
                            u[k][j][i].comp[c] = u[irhs][j][i].comp[c];
                        
                        u[k][j][i].comp[3] = -u[k][j][i].comp[3]; // Reflect z-momentum component
                    }
                }
            }
        }
    }
    
    // 2) Find the boundary extrapolated values of the conserved variables in all the cells 
    
    for ( k = zs-1; k < zs+zm+1; ++k ) {
        for ( j = ys-1; j < ys+ym+1; ++j ) { 
            for ( i = xs-1; i < xs+xm+1; ++i ) {
                
                local_k = k - (zs-1);
                local_j = j - (ys-1); 
                local_i = i - (xs-1); 
                
                for (c = 0; c < nVar; ++c) {
                    
                    for (oned_begin = -2; oned_begin < 3; ++oned_begin) {
                        u_x_loc[oned_begin+2] = u[k][j][i+oned_begin].comp[c];
                        u_y_loc[oned_begin+2] = u[k][j+oned_begin][i].comp[c];
                        u_z_loc[oned_begin+2] = u[k+oned_begin][j][i].comp[c];
                    }
                    
                    u_xy_loc[0] = u[k][j][i].comp[c];
                    u_xy_loc[1] = u[k][j+1][i+1].comp[c];
                    u_xy_loc[2] = u[k][j-1][i+1].comp[c];
                    u_xy_loc[3] = u[k][j+1][i-1].comp[c];
                    u_xy_loc[4] = u[k][j-1][i-1].comp[c];
                    
                    u_yz_loc[0] = u[k][j][i].comp[c];
                    u_yz_loc[1] = u[k+1][j+1][i].comp[c];
                    u_yz_loc[2] = u[k-1][j+1][i].comp[c];
                    u_yz_loc[3] = u[k+1][j-1][i].comp[c];
                    u_yz_loc[4] = u[k-1][j-1][i].comp[c];
                    
                    u_zx_loc[0] = u[k][j][i].comp[c];
                    u_zx_loc[1] = u[k+1][j][i+1].comp[c];
                    u_zx_loc[2] = u[k+1][j][i-1].comp[c];
                    u_zx_loc[3] = u[k-1][j][i+1].comp[c];
                    u_zx_loc[4] = u[k-1][j][i-1].comp[c];
                    
                    u_xyz_loc[0] = u[k+1][j+1][i+1].comp[c]; u_xyz_loc[1] = u[k+1][j+1][i-1].comp[c];
                    u_xyz_loc[2] = u[k+1][j-1][i+1].comp[c]; u_xyz_loc[3] = u[k-1][j+1][i+1].comp[c];
                    u_xyz_loc[4] = u[k+1][j-1][i-1].comp[c]; u_xyz_loc[5] = u[k-1][j+1][i-1].comp[c];
                    u_xyz_loc[6] = u[k-1][j-1][i+1].comp[c]; u_xyz_loc[7] = u[k-1][j-1][i-1].comp[c];
                    
                    weno(u_x_loc, u_y_loc, u_z_loc, u_xy_loc, u_yz_loc, u_zx_loc, u_xyz_loc, u_coeffs, coeffs);
                    
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
                    }
                }
            }
        }
    }

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
            
                for (c = 0; c < nVar; ++c)
                    Flux.comp[c] = 0.0; 
 
                for (f = 0; f < N_gp2d; ++f) {
                
                    for (c = 0; c < nVar; ++c) {
                        
                        QL.comp[c] = get_element_6d(Ctx->u_bnd, local_k, local_j, local_i-1, c, 1, f);
                        QR.comp[c] = get_element_6d(Ctx->u_bnd, local_k, local_j, local_i,   c, 0, f);
                        
                        for (iDim = 0; iDim < DIM; ++iDim) {
                        
                            grad_QL[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i-1, c, 1, f, iDim);
                            grad_QR[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i,   c, 0, f, iDim);
                        }
                    }
                
                    s_c = PDELLFRiemannSolver(&QL, &QR, nx, ny, nz, xloc, yloc, zloc, &Flux_conv); if (s_c>s_max_c) s_max_c = s_c;
                    s_v = PDEViscRiemannSolver(&QL, grad_QL, &QR, grad_QR, nx, ny, nz, &Flux_visc);    if (s_v>s_max_v) s_max_v = s_v;
                 
                
                    for (c = 0; c < nVar; ++c) {
                        Flux.comp[c] += w_gp2d[f]*(Flux_conv.comp[c] + Flux_visc.comp[c]);
                    }
                }
        
                for (c = 0; c < nVar; ++c) {
                    set_element_4d(Ctx->F, k-zs, j-ys, i-xs, c, Flux.comp[c]); 
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
            
                for (c = 0; c < nVar; ++c)
                    Flux.comp[c] = 0.0; 
 
                for (f = 0; f < N_gp2d; ++f) {
                
                    for (c = 0; c < nVar; ++c) {
                        
                        QL.comp[c] = get_element_6d(Ctx->u_bnd, local_k, local_j-1, local_i, c, 3, f);
                        QR.comp[c] = get_element_6d(Ctx->u_bnd, local_k, local_j, local_i,   c, 2, f);
                        
                        for (iDim = 0; iDim < DIM; ++iDim) {
                        
                            grad_QL[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j-1, local_i, c, 3, f, iDim);
                            grad_QR[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i,   c, 2, f, iDim);
                        }
                    }
                
                    s_c = PDELLFRiemannSolver(&QL, &QR, nx, ny, nz, xloc, yloc, zloc, &Flux_conv); if (s_c>s_max_c) s_max_c = s_c;
                    s_v = PDEViscRiemannSolver(&QL, grad_QL, &QR, grad_QR, nx, ny, nz, &Flux_visc);    if (s_v>s_max_v) s_max_v = s_v;
                 
                
                    for (c = 0; c < nVar; ++c) {
                        Flux.comp[c] += w_gp2d[f]*(Flux_conv.comp[c] + Flux_visc.comp[c]);
                    }
                }
        
                for (c = 0; c < nVar; ++c) {
                    set_element_4d(Ctx->G, k-zs, j-ys, i-xs, c, Flux.comp[c]); 
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
            
                for (c = 0; c < nVar; ++c)
                    Flux.comp[c] = 0.0; 
 
                for (f = 0; f < N_gp2d; ++f) {
                
                    for (c = 0; c < nVar; ++c) {
                        
                        QL.comp[c] = get_element_6d(Ctx->u_bnd, local_k-1, local_j, local_i, c, 5, f);
                        QR.comp[c] = get_element_6d(Ctx->u_bnd, local_k, local_j, local_i,   c, 4, f);
                        
                        for (iDim = 0; iDim < DIM; ++iDim) {
                        
                            grad_QL[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k-1, local_j, local_i, c, 5, f, iDim);
                            grad_QR[c][iDim] = get_element_7d(Ctx->u_bnd_grad, local_k, local_j, local_i,   c, 4, f, iDim);
                        }
                    }
                
                    s_c = PDELLFRiemannSolver(&QL, &QR, nx, ny, nz, xloc, yloc, zloc, &Flux_conv);  if (s_c>s_max_c) s_max_c = s_c;
                    s_v = PDEViscRiemannSolver(&QL, grad_QL, &QR, grad_QR, nx, ny, nz, &Flux_visc); if (s_v>s_max_v) s_max_v = s_v;
                 
                
                    for (c = 0; c < nVar; ++c) {
                        Flux.comp[c] += w_gp2d[f]*(Flux_conv.comp[c] + Flux_visc.comp[c]);
                    }
                }
        
                for (c = 0; c < nVar; ++c) {
                    set_element_4d(Ctx->H, k-zs, j-ys, i-xs, c, Flux.comp[c]); 
                }
            }
        }
    }
    
    // 4) Find the rhs in each cell 

    for(k=zs; k<zs+zm; ++k) {
        for (j=ys; j<ys+ym; ++j) {
            for (i=xs; i<xs+xm; ++i) {

                for (c = 0 ; c < nVar; ++c) {
        
                    rhs[k][j][i].comp[c]  = -r1_h*(get_element_4d(Ctx->F, k-zs, j-ys, i+1-xs, c) - get_element_4d(Ctx->F, k-zs, j-ys, i-xs, c))
                                            -r1_h*(get_element_4d(Ctx->G, k-zs, j+1-ys, i-xs, c) - get_element_4d(Ctx->G, k-zs, j-ys, i-xs, c))
                                            -r1_h*(get_element_4d(Ctx->H, k+1-zs, j-ys, i-xs, c) - get_element_4d(Ctx->H, k-zs, j-ys, i-xs, c));
                }
            }
        }
    }
    
    ierr = DMDAVecRestoreArray(da,Ctx->localU,&u);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,RHS,&rhs);CHKERRQ(ierr);

    dt = (1.0/3.0)*(Ctx->CFL*Ctx->h)/(s_max_c + (r1_h*s_max_v)*2.0); // 1/3 is for three dimensions

    ierr = MPI_Allreduce(&dt, &Ctx->dt, 1, MPIU_REAL,MPIU_MIN, PetscObjectComm((PetscObject)da));CHKERRQ(ierr);

    return ierr; 
}
