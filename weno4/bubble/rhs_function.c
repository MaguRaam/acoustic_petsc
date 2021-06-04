/*
 * rhs_function.c
 *      Author: sunder
 */ 
#include "hype.h" 

//----------------------------------------------------------------------------
// Compute the value of RHS for each cell in the domain using 
// primitive variables
//----------------------------------------------------------------------------

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec RHS, void* ctx) {

    PetscErrorCode ierr;           
    AppCtx *Ctx = (AppCtx*)ctx; 
    DM da;                         
    PetscInt c,i,j,k,f,q,xs,ys,xm,ym,xs_g,ys_g,xm_g,ym_g,oned_begin,oned_end,irhs;                
    Field   **w;                   
    Field   **rhs;                 
    PetscReal s_c, s_max_c = 0.0;
    PetscReal w_x_loc[s_width], w_y_loc[s_width], w_xy_loc[s_width];  
    PetscReal dt; 
    PetscReal coeffs[nDOF], sol[nDOF][nVar];
    PetscReal value; 
    PetscReal r1_h = 1./(Ctx->h); 
    PetscReal x_loc, y_loc, nx, ny; 
    PetscInt local_i, local_j;
    PetscBool PAD; 
    
    PetscReal VNode[N_node][nVar], V[nVar];
    PetscReal VL[nVar], VR[nVar];
    PetscReal Flux_conv[6], Flux[6];

    ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
    
    // Calculate primitive variables in each cell 
    
    ierr = ComputePrimitiveVariables(U, Ctx->W, da);CHKERRQ(ierr);
    
    // Scatter global->local to have access to the required ghost values 

    ierr = DMGlobalToLocalBegin(da, Ctx->W, INSERT_VALUES, Ctx->localU);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Ctx->W, INSERT_VALUES,   Ctx->localU);CHKERRQ(ierr);

    // Read the local solution to the array u  

    ierr = DMDAVecGetArrayRead(da, Ctx->localU, &w); CHKERRQ(ierr); 
    ierr = DMDAVecGetArray(da, RHS, &rhs);CHKERRQ(ierr);

    ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
    ierr = DMDAGetGhostCorners(da, &xs_g, &ys_g, NULL, &xm_g, &ym_g, NULL);

    //--------------------------------------------------------------
    // Apply Boundary Conditions 
    //--------------------------------------------------------------

    oned_begin = 0; oned_end = Ctx->N_x-1; 

    for (j = ys_g; j < ys_g+ym_g; ++j) {
        for (i = xs_g; i < xs_g+xm_g; ++i) {
            
            if (i < 0)  { // Left boundary 
                
                // Transmissive/Outflow Boundary 
                
                if (Ctx->left_boundary == transmissive) {
                    
                    irhs = oned_begin; 
                
                    for (c = 0; c < nVar; ++c)
                        w[j][i].comp[c] = w[j][irhs].comp[c];
                }
                
                // Reflective Boundary 
                
                if (Ctx->left_boundary == reflective) {
                    
                    irhs = oned_begin - 1 - i; 
                
                    for (c = 0; c < nVar; ++c)
                        w[j][i].comp[c] = w[j][irhs].comp[c];
                    
                    w[j][i].comp[1] = -w[j][i].comp[1]; // Reflect x-momentum component 
                }
                
            }
            
            if (i >= Ctx->N_x) { // Right Boundary 
                
                // Transmissive/Outflow Boundary 
                
                if (Ctx->right_boundary == transmissive) {
                    
                    irhs = oned_end; 
                    
                    for (c = 0; c < nVar; ++c)
                        w[j][i].comp[c] = w[j][irhs].comp[c];
                }
                
                // Reflective Boundary
                
                if (Ctx->right_boundary == reflective) {
                    
                    irhs = 2*oned_end - i + 1; 
                    
                    for (c = 0; c < nVar; ++c)
                        w[j][i].comp[c] = w[j][irhs].comp[c];
                    
                    w[j][i].comp[1] = -w[j][i].comp[1]; // Reflect x-momentum component 
                }
            }
        }
    }

    oned_begin = 0; oned_end = Ctx->N_y-1;

    for (j = ys_g; j < ys_g+ym_g; ++j) {
        for (i = xs_g; i < xs_g+xm_g; ++i) {
            
            if (j < 0) { // Bottom boundary 
                
                // Transmissive/Outflow Boundary
                
                if (Ctx->bottom_boundary == transmissive) {
                
                    irhs = oned_begin;
                    
                    for (c = 0; c < nVar; ++c)
                        w[j][i].comp[c] = w[irhs][i].comp[c]; 
                }
                
                // Reflective Boundary
                
                if (Ctx->bottom_boundary == reflective) {
                
                    irhs = oned_begin - 1 - j;
                    
                    for (c = 0; c < nVar; ++c)
                        w[j][i].comp[c] = w[irhs][i].comp[c]; 
                     
                    w[j][i].comp[2] = -w[j][i].comp[2]; // Reflect y-momentum component 
                }
            }
            
            if (j >= Ctx->N_y) { // Top boundary 
                
                // Transmissive/Outflow Boundary
                
                if (Ctx->top_boundary == transmissive) {
                
                    irhs = oned_end;
                    
                    for (c = 0; c < nVar; ++c)
                        w[j][i].comp[c] = w[irhs][i].comp[c];
                }
                
                // Reflective Boundary
                
                if (Ctx->top_boundary == reflective) {
                
                    irhs = 2*oned_end - j + 1;;
                    
                    for (c = 0; c < nVar; ++c)
                        w[j][i].comp[c] = w[irhs][i].comp[c];
                
                    w[j][i].comp[2] = -w[j][i].comp[2]; // Reflect y-momentum component 
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
            PAD = PETSC_TRUE; 
            
            // Select stencils for each component of the solution 
        
            for (c = 0 ; c < nVar; ++c) {
                
                w_x_loc[0] = w[j][i-2].comp[c]; 
                w_x_loc[1] = w[j][i-1].comp[c]; 
                w_x_loc[2] = w[j][i].comp[c]; 
                w_x_loc[3] = w[j][i+1].comp[c]; 
                w_x_loc[4] = w[j][i+2].comp[c];
                
                w_y_loc[0] = w[j-2][i].comp[c]; 
                w_y_loc[1] = w[j-1][i].comp[c]; 
                w_y_loc[2] = w[j][i].comp[c]; 
                w_y_loc[3] = w[j+1][i].comp[c]; 
                w_y_loc[4] = w[j+2][i].comp[c];

                w_xy_loc[0] = w[j][i].comp[c];
                w_xy_loc[1] = w[j+1][i+1].comp[c];
                w_xy_loc[2] = w[j-1][i+1].comp[c];
                w_xy_loc[3] = w[j+1][i-1].comp[c];
                w_xy_loc[4] = w[j-1][i-1].comp[c];
                
                weno(w_x_loc, w_y_loc, w_xy_loc, coeffs);
                
                // Store the coefficients 
                
                for (k = 0; k < nDOF; ++k) {
                    sol[k][c] = coeffs[k]; 
                }
                
                // Evaluate solution at various nodes to check physical admissibility of the solution  
                
                for (q = 0; q < N_node; ++q) {
                    
                    VNode[q][c] = 0.0;
                    
                    for (k = 0; k < nDOF; ++k)
                        VNode[q][c] += coeffs[k]*get_element_2d(Ctx->phiNode,q,k);
                }
            }
            
            // Check physical admissibility of the solution  and reduce to TVD if necessary 
            
            for (q = 0; q < N_node; ++q) {
                
                for (c = 0; c < nVar; ++c) {
                    V[c] = VNode[q][c];
                }
                
                PAD = PDECheckPADPrim(V);
                    
                if (PAD == PETSC_FALSE)
                    break; 
            }
            
            
            if (PAD == PETSC_FALSE) {
                
                
                for (c = 0 ; c < nVar; ++c) {
                    
                    sol[0][c] = w[j][i].comp[c];
                    sol[1][c] = minmod(w[j][i+1].comp[c]-w[j][i].comp[c],w[j][i].comp[c]-w[j][i-1].comp[c]);
                    sol[2][c] = minmod(w[j+1][i].comp[c]-w[j][i].comp[c],w[j][i].comp[c]-w[j-1][i].comp[c]);
                    
                    for (k = 3; k < nDOF; ++k) {
                        sol[k][c] = 0.0; 
                    }
                }
            
            }
            
            // Find the values of conserved variables at face quadrature points 
            
            for (c = 0 ; c < nVar; ++c) {
            
                // Get coefficients 
                
                for (f = 0; f < 4; ++f) {
                    for (q = 0; q < N_gp2; ++q) {
                        
                        value = 0.0; 
                        
                        for (k = 0; k < nDOF; ++k)
                            value += sol[k][c]*get_element_3d(Ctx->phiFace,f,q,k);
                        
                        set_element_5d(Ctx->u_bnd, local_j, local_i, c, f, q, value);
                    
                    }
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
            
            x_loc = Ctx->x_min + (PetscReal)(i)*Ctx->h;
            y_loc = Ctx->y_min + (PetscReal)(j)*Ctx->h;
            
            for (c = 0; c < 6; ++c) {
                Flux[c] = 0.0;
            }

            for (q = 0; q < N_gp2; ++q) {
                
                for (c = 0; c < nVar; ++c) {
                    VL[c] = get_element_5d(Ctx->u_bnd, local_j, local_i-1, c, 1, q);
                    VR[c] = get_element_5d(Ctx->u_bnd, local_j, local_i,   c, 0, q);
                }
                
                s_c = rotHLLCRiemannSolver(VL, VR, nx, ny, x_loc, y_loc, Flux_conv); if (s_c>s_max_c) s_max_c = s_c; 
                
                for (c = 0; c < 6; ++c) {
                    Flux[c] += w_gp2[q]*Flux_conv[c];
                }
            }
        
            for (c = 0; c < 6; ++c) {
                set_element_3d(Ctx->F, j-ys, i-xs, c, Flux[c]); 
            }
        }
    }
            
    // Find the upwind flux on each face in y-dirction 
    
    nx = 0.0; ny = 1.0;

    for (j = ys; j < ys+ym+1; ++j) {
        for (i = xs; i < xs+xm; ++i) {
            
            local_j = j - (ys-1); 
            local_i = i - (xs-1); 
            
            x_loc = Ctx->x_min + (PetscReal)(i)*Ctx->h;
            y_loc = Ctx->y_min + (PetscReal)(j)*Ctx->h;
            
            for (c = 0; c < 6; ++c) {
                Flux[c] = 0.0; 
            }

            for (q = 0; q < N_gp2; ++q) {
                
                for (c = 0; c < nVar; ++c) {
                    VL[c] = get_element_5d(Ctx->u_bnd, local_j-1, local_i, c, 3, q);
                    VR[c] = get_element_5d(Ctx->u_bnd, local_j,   local_i, c, 2, q);
                }
                
                s_c = rotHLLCRiemannSolver(VL, VR, nx, ny, x_loc, y_loc, Flux_conv); if (s_c>s_max_c) s_max_c = s_c; 
                
                for (c = 0; c < 6; ++c) {
                    Flux[c] += w_gp2[q]*Flux_conv[c];
                }
            }
            
            for (c = 0; c < 6; ++c) {
                set_element_3d(Ctx->G, j-ys, i-xs, c, Flux[c]); 
            }
        }
    }

    // Now find the rhs in each cell 

    for (j=ys; j<ys+ym; ++j) {
        for (i=xs; i<xs+xm; ++i) {
            
            for (c = 0 ; c < nVar; ++c) {
                
                if (c == 4) {
                
                    rhs[j][i].comp[c]  = -r1_h*((get_element_3d(Ctx->F, j-ys, i+1-xs, c) - 
                                                 w[j][i].comp[c]*get_element_3d(Ctx->F, j-ys, i+1-xs, c+1))
                                              -(get_element_3d(Ctx->F, j-ys, i-xs, c) - 
                                                 w[j][i].comp[c]*get_element_3d(Ctx->F, j-ys, i-xs, c+1)) )
                    
                                        -r1_h*((get_element_3d(Ctx->G, j+1-ys, i-xs, c) - 
                                                w[j][i].comp[c]*get_element_3d(Ctx->G, j+1-ys, i-xs, c+1))
                                             -(get_element_3d(Ctx->G, j-ys, i-xs, c) - 
                                                w[j][i].comp[c]*get_element_3d(Ctx->G, j-ys, i-xs, c+1)) );
                }
                
                else {
    
                    rhs[j][i].comp[c]  = -r1_h*(get_element_3d(Ctx->F, j-ys, i+1-xs, c) - get_element_3d(Ctx->F, j-ys, i-xs, c))
                                         -r1_h*(get_element_3d(Ctx->G, j+1-ys, i-xs, c) - get_element_3d(Ctx->G, j-ys, i-xs, c));
                }
                    
            }
            
        }
    }

    ierr = DMDAVecRestoreArray(da,Ctx->localU,&w);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,RHS,&rhs);CHKERRQ(ierr);

    dt = (Ctx->CFL*Ctx->h)/(2.0*s_max_c); // 2.0 corresponds to 2 dimensions 

    ierr = MPI_Allreduce(&dt, &Ctx->dt, 1, MPIU_REAL,MPIU_MIN, PetscObjectComm((PetscObject)da));CHKERRQ(ierr);

    return ierr; 
}

