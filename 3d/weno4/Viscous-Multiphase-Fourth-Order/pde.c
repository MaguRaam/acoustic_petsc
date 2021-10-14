#include "hype.h"

/* Definitions of various PDE functions */

//----------------------------------------------------------------------------
// Convert a conserved variable to primitive variable
//----------------------------------------------------------------------------

void PDECons2Prim(const Field* Q, Field* V) {
    
    PetscReal g = 1.0 + (g1-1.0)*(g2-1.0)/((1.0-Q->comp[5])*(g1 -1.0) + Q->comp[5]*(g2-1.0));
    PetscReal P_inf = ((g -1.0)/g)*( g1*p1*Q->comp[5]/(g1 - 1.0) + g2*p2*(1.0 - Q->comp[5])/(g2 - 1.0) );
    PetscReal temp = 1.0/Q->comp[0]; 
    
    V->comp[0] = Q->comp[0];
    V->comp[1] = Q->comp[1]*temp;
    V->comp[2] = Q->comp[2]*temp;
    V->comp[3] = Q->comp[3]*temp; 
    V->comp[4] = (g -1.0)*( Q->comp[4] - 0.5*temp*(Q->comp[1]*Q->comp[1] + Q->comp[2]*Q->comp[2] + Q->comp[3]*Q->comp[3]) )  - g*P_inf;
    V->comp[5] = Q->comp[5];
}

//----------------------------------------------------------------------------
// Convert a primitive variable to conserved variable
//----------------------------------------------------------------------------

void PDEPrim2Cons(const Field* V, Field* Q) {
    
    PetscReal g = 1.0 + (g1-1.0)*(g2-1.0)/((1.0-V->comp[5])*(g1 -1.0) + V->comp[5]*(g2-1.0));
    PetscReal P_inf = ((g -1.0)/g)*( g1*p1*V->comp[5]/(g1 - 1.0) + g2*p2*(1.0 - V->comp[5])/(g2 - 1.0) );

    PetscReal e = (V->comp[4] + g*P_inf)/(g - 1.0);
    PetscReal k = 0.5*V->comp[0]*(V->comp[1]*V->comp[1] + V->comp[2]*V->comp[2] + V->comp[3]*V->comp[3]);

    Q->comp[0] = V->comp[0];
    Q->comp[1] = V->comp[0]*V->comp[1];
    Q->comp[2] = V->comp[0]*V->comp[2];
    Q->comp[3] = V->comp[0]*V->comp[3]; 
    Q->comp[4] = k + e;
    Q->comp[5] = V->comp[5];
}

//----------------------------------------------------------------------------
// Find the conservative flux components F in the given normal direction
//----------------------------------------------------------------------------

PetscReal PDEConsFlux (const Field* Q, 
                       const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                       const PetscReal x,  const PetscReal y,  const PetscReal z, 
                       Field* F) {


    PetscReal g = 1.0 + (g1-1.0)*(g2-1.0)/((1.0-Q->comp[5])*(g1 -1.0) + Q->comp[5]*(g2-1.0));
    PetscReal P_inf = ((g -1.0)/g)*( g1*p1*Q->comp[5]/(g1 - 1.0) + g2*p2*(1.0 - Q->comp[5])/(g2 - 1.0) );
    PetscReal temp = 1.0/Q->comp[0]; 
    PetscReal rho = Q->comp[0];
    PetscReal u = Q->comp[1]*temp;
    PetscReal v = Q->comp[2]*temp;
    PetscReal w = Q->comp[3]*temp;
    PetscReal E = Q->comp[4]; 
    PetscReal p = (g -1.0)*( Q->comp[4] - 0.5*rho*(u*u + v*v + w*w) )  - g*P_inf;
    
    // Check if the input state is physically admissible 

    if (rho < rho_floor) {
        printf("Negative density = %f\n", rho);
        printf("At x = %f, y = %f\n", x, y);  
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    if ((p + P_inf)  < prs_floor) {
        printf("Negative pressure, p = %f\n", p + P_inf);
        printf("At x = %f, y = %f\n", x, y);  
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    // Now find the fluxes 
    
    F->comp[0] = nx*rho*u         + ny*rho*v         + nz*rho*w;
    F->comp[1] = nx*(rho*u*u + p) + ny*rho*u*v       + nz*rho*u*w;
    F->comp[2] = nx*rho*u*v       + ny*(rho*v*v + p) + nz*rho*v*w;
    F->comp[3] = nx*rho*u*w       + ny*rho*v*w       + nz*(rho*w*w + p);
    F->comp[4] = (E + p)*(nx*u + ny*v + nz*w);
    F->comp[5] = 0.0; 

    // Also obtain the maximum eigen value 

    PetscReal a = PetscSqrtReal(g*(p+P_inf)/rho); 

    // Also obtain the maximum eigen value 

    PetscReal s_max = PetscAbsReal(u*nx + v*ny + w*nz) + a;

    return s_max;
}

//----------------------------------------------------------------------------
// Find the Viscous flux in the normal direction F_v.n 
//----------------------------------------------------------------------------

PetscReal PDEViscFluxPrim(const Field* W, PetscReal grad_V[nVar][DIM], PetscReal nx, PetscReal ny, PetscReal nz, Field* F) {
    
    const PetscReal r2_3 = 2./3;
    const PetscReal r4_3 = 4./3;
    
    PetscReal rho = W->comp[0]; 
    PetscReal u   = W->comp[1];
    PetscReal v   = W->comp[2];
    PetscReal w   = W->comp[3];
    PetscReal phi = W->comp[5];

    PetscReal mu = phi*MU_1 + (1.0-phi)*MU_2;     // Viscosity 
    PetscReal mu_b = -r2_3*mu;
    
    // Viscous flux 

    PetscReal div_v = grad_V[1][0] + grad_V[2][1] + grad_V[3][2]; 
    
    PetscReal tau_xx = mu_b*(div_v) + 2.0*mu*(grad_V[1][0]);
    PetscReal tau_yy = mu_b*(div_v) + 2.0*mu*(grad_V[2][1]);
    PetscReal tau_zz = mu_b*(div_v) + 2.0*mu*(grad_V[3][2]);
    
    PetscReal tau_xy = mu*(grad_V[1][1] + grad_V[2][0]);
    PetscReal tau_yz = mu*(grad_V[2][2] + grad_V[3][1]);
    PetscReal tau_xz = mu*(grad_V[1][2] + grad_V[3][0]);
    
    F->comp[0] =  0.0;
    F->comp[1] = -nx*tau_xx - ny*tau_xy - nz*tau_xz;
    F->comp[2] = -nx*tau_xy - ny*tau_yy - nz*tau_yz;
    F->comp[3] = -nx*tau_xz - ny*tau_yz - nz*tau_zz; 
    F->comp[4] = -nx*(u*tau_xx + v*tau_xy + w*tau_xz) - ny*(u*tau_xy + v*tau_yy + w*tau_yz) - nz*(u*tau_xz + v*tau_yz + w*tau_zz);
    F->comp[5] =  0.0; 
        
    return r4_3*mu/rho; 
}

//----------------------------------------------------------------------------
// Find the conservative flux components F in the given normal direction. But
// the input to the function is a primitive variable
//----------------------------------------------------------------------------

PetscReal PDEConsFluxPrim (const Field* V, 
                           const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                           const PetscReal x,  const PetscReal y,  const PetscReal z, 
                            PetscReal F[7]) {


    PetscReal phi = V->comp[5];
    PetscReal g = 1.0 + (g1-1.0)*(g2-1.0)/((1.0-phi)*(g1 -1.0) + phi*(g2-1.0));
    PetscReal P_inf = ((g -1.0)/g)*(g1*p1*phi/(g1 - 1.0) + g2*p2*(1.0 - phi)/(g2 - 1.0));
    
    PetscReal rho = V->comp[0];
    PetscReal u = V->comp[1];
    PetscReal v = V->comp[2];
    PetscReal w = V->comp[3];
    PetscReal p = V->comp[4];
    
    PetscReal e = (V->comp[4] + g*P_inf)/(g - 1.0);
    PetscReal k = 0.5*rho*(u*u + v*v + w*w);
    
    PetscReal E = k + e;
    PetscReal un = nx*u + ny*v + nz*w;
    
    
    // Check if the input state is physically admissible 

    if (rho < rho_floor) {
        printf("Negative density = %f\n", rho);
        printf("At x = %f, y = %f\n", x, y);  
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    if ((p + P_inf)  < prs_floor) {
        printf("Negative pressure, p = %f\n", p + P_inf);
        printf("At x = %f, y = %f\n", x, y);  
        MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    // Now find the fluxes 

    F[0] = rho*un;
    F[1] = rho*u*un + p*nx;
    F[2] = rho*v*un + p*ny;
    F[3] = rho*w*un + p*nz;
    F[4] = (E + p)*un;
    F[5] = phi*un;
    F[6] = 0.0; 
    
    // Also obtain the maximum eigen value 

    PetscReal a = PetscSqrtReal(g*(p+P_inf)/rho); 

    // Also obtain the maximum eigen value 

    PetscReal s_max = PetscAbsReal(un) + a;

    return s_max;
}

//----------------------------------------------------------------------------
// Rusanov/LLF Riemann Solver 
//----------------------------------------------------------------------------

PetscReal PDELLFRiemannSolver(const Field* QL, const Field* QR, 
                              const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                              const PetscReal x, const PetscReal y,  const PetscReal z, 
                              Field* Flux) {
        
    Field FL, FR; PetscInt c; 

    PetscReal s_max_l = PDEConsFlux(QL, nx, ny, nz, x, y, z, &FL); 
    PetscReal s_max_r = PDEConsFlux(QR, nx, ny, nz, x, y, z, &FR);

    PetscReal s_max = PetscMax(s_max_l, s_max_r);

    for (c = 0; c < nVar; ++c) {
        Flux->comp[c] = 0.5*(FR.comp[c] + FL.comp[c] - s_max*(QR->comp[c] - QL->comp[c]));
    }

    return s_max; 
}

//----------------------------------------------------------------------------
// Check Physical admissibility 
//----------------------------------------------------------------------------

PetscBool PDECheckPAD(const Field V) {
        
    PetscBool PAD = PETSC_TRUE; 
    
    PetscReal phi = V.comp[5];
    PetscReal g = 1.0 + (g1-1.0)*(g2-1.0)/((1.0-phi)*(g1 -1.0) + phi*(g2-1.0));
    PetscReal P_inf = ((g -1.0)/g)*(g1*p1*phi/(g1 - 1.0) + g2*p2*(1.0 - phi)/(g2 - 1.0));
    
    PetscReal rho = V.comp[0];
    PetscReal p = V.comp[4];
    
    if (rho < rho_floor) {
        PAD = PETSC_FALSE;
    }
    
    if (p+P_inf < prs_floor) {
        PAD = PETSC_FALSE;
    }
    
    return PAD;
}

//----------------------------------------------------------------------------
// HLLC Riemann Solver 
//----------------------------------------------------------------------------

// Simple function to invert a 3x3 system 

void invert3x3matrix(const PetscReal M[3][3], PetscReal iM[3][3]) {
    const PetscReal t4  = M[0][0]*M[1][1], t6  = M[0][0]*M[1][2],
                    t8  = M[0][1]*M[1][0], t00 = M[0][2]*M[1][0],
                    t01 = M[0][1]*M[2][0], t04 = M[0][2]*M[2][0],
                    t07 = 1.0/(t4*M[2][2] - t6*M[2][1] - t8*M[2][2] + t00*M[2][1] + t01*M[1][2] - t04*M[1][1]);
    
    iM[0][0] = (M[1][1]*M[2][2] - M[1][2]*M[2][1])*t07;
    iM[0][1] = -(M[0][1]*M[2][2] - M[0][2]*M[2][1])*t07;
    iM[0][2] = -(-M[0][1]*M[1][2] + M[0][2]*M[1][1])*t07;
    iM[1][0] = -(M[1][0]*M[2][2] - M[1][2] * M[2][0])*t07;
    iM[1][1] = (M[0][0]*M[2][2] - t04)*t07;
    iM[1][2] = -(t6 - t00)*t07;
    iM[2][0] = -(-M[1][0]*M[2][1] + M[1][1]*M[2][0])*t07;
    iM[2][1] = -(M[0][0]*M[2][1] - t01)*t07;
    iM[2][2] = (t4 - t8)*t07;
    
}

PetscReal PDEHLLCRiemannSolver(const Field* VL, const Field* VR, 
                               const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                               const PetscReal x, const PetscReal y,  const PetscReal z, 
                               PetscReal Flux[7]) {
        
    PetscInt i ;
    PetscReal rho_L, rho_R, u_L, u_R, v_L, v_R, w_L, w_R, P_L, P_R, c_L, c_R, E_L, E_R ;
    PetscReal un_L, un_R, ut1_L, ut1_R, ut2_L, ut2_R ;
    PetscReal t1x, t1y, t1z, t2x, t2y, t2z ;
    PetscReal un, ut1, ut2 ;
    PetscReal S_L, S_R, S_star ;
    Field QL, QR, QL_star, QR_star; 
    PetscReal FL[7], FR[7], FL_star[7], FR_star[7];
    PetscReal Rot_Mat[3][3], InvRot_Mat[3][3], Local;

    rho_L = VL->comp[0]; u_L = VL->comp[1]; v_L = VL->comp[2]; w_L = VL->comp[3]; P_L = VL->comp[4];
    rho_R = VR->comp[0]; u_R = VR->comp[1]; v_R = VR->comp[2]; w_R = VR->comp[3]; P_R = VR->comp[4];
    
    PDEPrim2Cons(VL, &QL); PDEPrim2Cons(VR, &QR);
    
    PetscReal smaxl = PDEConsFluxPrim (VL, nx, ny, nz, x, y, z, FL); 
    PetscReal smaxr = PDEConsFluxPrim (VR, nx, ny, nz, x, y, z, FR); 
            
    // Define the two tangent directions following Miller and Colella's JCP paper
    
    if(PetscAbsReal(ny + nz) >  PetscAbsReal(ny - nz) ) {

            Local = PetscSqrtReal( 2.0*(1.0 + nz*(ny - nx) + nx*ny) ) ;
            t1x = (ny + nz)/Local ; t1y = (nz - nx)/Local ; t1z = -(nx + ny)/Local ;
            t2x = ( nx*(nz - ny) - ny*ny - nz*nz )/Local ; 
            t2y = ( ny*(nx + nz) + nx*nx + nz*nz )/Local ; 
            t2z = ( nz*(nx - nz) - nx*nx - ny*ny )/Local ; 

    } 
    
    else {

            Local = PetscSqrtReal( 2.0*(1.0 - nx*ny - nx*nz - ny*nz) ) ;
            t1x = (ny - nz)/Local ; t1y = (nz - nx)/Local ; t1z = (nx - ny)/Local ;
            t2x = ( nx*(ny + nz) - ny*ny - nz*nz )/Local ; 
            t2y = ( ny*(nx + nz) - nx*nx - nz*nz )/Local ;
            t2z = ( nz*(nx + ny) - nx*nx - ny*ny )/Local ;  
    }

    Rot_Mat[0][0] = nx ; Rot_Mat[0][1] = ny ; Rot_Mat[0][2] = nz ;
    Rot_Mat[1][0] = t1x ; Rot_Mat[1][1] = t1y ; Rot_Mat[1][2] = t1z ;
    Rot_Mat[2][0] = t2x ; Rot_Mat[2][1] = t2y ; Rot_Mat[2][2] = t2z ;

    invert3x3matrix(Rot_Mat, InvRot_Mat);

    un_L = u_L*nx + v_L*ny + w_L*nz ; ut1_L = u_L*t1x + v_L*t1y + w_L*t1z ; ut2_L = u_L*t2x + v_L*t2y + w_L*t2z ;
    un_R = u_R*nx + v_R*ny + w_R*nz ; ut1_R = u_R*t1x + v_R*t1y + w_R*t1z ; ut2_R = u_R*t2x + v_R*t2y + w_R*t2z ;

    c_L = smaxl - PetscAbsReal(un_L); c_R = smaxr - PetscAbsReal(un_R); 

    E_L = QL.comp[4];
    E_R = QR.comp[4];

    S_L = PetscMin((un_R - c_R), (un_L - c_L)) ;
    S_R = PetscMax((un_L + c_L), (un_R + c_R)) ;
    S_star = ( P_R - P_L + rho_L*un_L*(S_L-un_L) - rho_R*un_R*(S_R - un_R) )/(rho_L*(S_L - un_L) - rho_R*(S_R - un_R)) ;

    FL[6] = un_L; FR[6] = un_R;


    QL_star.comp[0] = rho_L*(S_L - un_L)/(S_L - S_star);
    un = S_star; ut1 = ut1_L; ut2 = ut2_L;
    QL_star.comp[1] = QL_star.comp[0]*(un*InvRot_Mat[0][0] + ut1*InvRot_Mat[0][1] + ut2*InvRot_Mat[0][2]);
    QL_star.comp[2] = QL_star.comp[0]*(un*InvRot_Mat[1][0] + ut1*InvRot_Mat[1][1] + ut2*InvRot_Mat[1][2]);
    QL_star.comp[3] = QL_star.comp[0]*(un*InvRot_Mat[2][0] + ut1*InvRot_Mat[2][1] + ut2*InvRot_Mat[2][2]);
    QL_star.comp[4] = QL_star.comp[0]*( (E_L/rho_L) + (S_star - un_L)*(S_star + P_L/(rho_L*(S_L - un_L)) ) );
    QL_star.comp[5] = VL->comp[5]*(S_L - un_L)/(S_L - S_star);

    QR_star.comp[0] = rho_R*(S_R - un_R)/(S_R - S_star);
    un = S_star; ut1 = ut1_R; ut2 = ut2_R;
    QR_star.comp[1] = QR_star.comp[0]*(un*InvRot_Mat[0][0] + ut1*InvRot_Mat[0][1] + ut2*InvRot_Mat[0][2]) ;
    QR_star.comp[2] = QR_star.comp[0]*(un*InvRot_Mat[1][0] + ut1*InvRot_Mat[1][1] + ut2*InvRot_Mat[1][2]) ;
    QR_star.comp[3] = QR_star.comp[0]*(un*InvRot_Mat[2][0] + ut1*InvRot_Mat[2][1] + ut2*InvRot_Mat[2][2]) ;
    QR_star.comp[4] = QR_star.comp[0]*( (E_R/rho_R) + (S_star - un_R)*(S_star + P_R/(rho_R*(S_R - un_R)) ) ) ;
    QR_star.comp[5] = VR->comp[5]*(S_R - un_R)/(S_R - S_star) ;

    
    for(i = 0 ; i < 6 ; i++) 
        FL_star[i] = FL[i] + S_L*(QL_star.comp[i] - QL.comp[i]); 
    
    FL_star[6] = un_L + S_L*( ((S_L - un_L)/(S_L - S_star)) - 1.0 );


    for(i = 0 ; i < 6 ; i++) 
        FR_star[i] = FR[i] + S_R*(QR_star.comp[i] - QR.comp[i]); 
    
    FR_star[6] = un_R + S_R*( ((S_R - un_R)/(S_R - S_star)) - 1.0 ) ;

    if( S_L > 0.0 ) {
            for(i = 0 ; i < 7 ; i++) Flux[i] = FL[i];
    } else if((S_star >= 0.0) && (S_L < 0.0)) {
            for(i = 0 ; i < 7 ; i++) Flux[i] = FL_star[i];
    } else if((S_star < 0.0) && (S_R >= 0.0)) {
            for(i = 0 ; i < 7 ; i++) Flux[i] = FR_star[i];
    } else if(S_R < 0.0) {
            for(i = 0 ; i < 7 ; i++) Flux[i] = FR[i];
    }
    
    return PetscMax(smaxl, smaxr);
}

//----------------------------------------------------------------------------
// Rotated HLLC Riemann Solver (adds diffusion along the tangential direction)
//----------------------------------------------------------------------------

PetscReal PDErotHLLCRiemannSolver(const Field* VL, const Field* VR, 
                               const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                               const PetscReal x, const PetscReal y,  const PetscReal z, 
                               PetscReal Flux[7]) {
    
    PetscReal Flux1[7], Flux2[7], Flux3[7]; 

    PetscInt c;
    PetscReal alpha1, alpha2, alpha3, n1x, n1y, n1z, t1x, t1y, t1z, t2x, t2y, t2z, u_L, u_R, v_L, v_R, w_L, w_R, du, dv, dw, dq, Local;


    u_L = VL->comp[1]; v_L = VL->comp[2]; w_L = VL->comp[3];
    u_R = VR->comp[1]; v_R = VR->comp[2]; w_R = VR->comp[3];
    du = u_R - u_L ; dv = v_R - v_L ; dw = w_R - w_L ;
    dq = PetscSqrtReal(du*du + dv*dv + dw*dw) ;

    if(dq < 1.0E-10) { n1x = nx ; n1y = ny ; n1z = nz;}
    else { n1x = du/dq ; n1y = dv/dq ; n1z = dw/dq ;}

    alpha1 = (n1x*nx + n1y*ny + n1z*nz) ;
    if(alpha1 < 0.0) { n1x = -n1x ; n1y = -n1y ; n1z = -n1z ; alpha1 = -alpha1 ; }

    if( PetscAbsReal(n1y + n1z) >  PetscAbsReal(n1y - n1z) ) {
        Local = PetscSqrtReal( 2.0*(1.0 + n1z*(n1y - n1x) + n1x*n1y) ) ;
        t1x = (n1y + n1z)/Local ; t1y = (n1z - n1x)/Local ; t1z = -(n1x + n1y)/Local ;
        t2x = ( n1x*(n1z - n1y) - n1y*n1y - n1z*n1z )/Local ; 
        t2y = ( n1y*(n1x + n1z) + n1x*n1x + n1z*n1z )/Local ; 
        t2z = ( n1z*(n1x - n1y) - n1x*n1x - n1y*n1y )/Local ; 
    } 
    
    else {

        Local = PetscSqrtReal( 2.0*(1.0 - n1x*n1y - n1x*n1z - n1y*n1z) ) ;
        t1x = (n1y - n1z)/Local ; t1y = (n1z - n1x)/Local ; t1z = (n1x - n1y)/Local ;
        t2x = ( n1x*(n1y + n1z) - n1y*n1y - n1z*n1z )/Local ; 
        t2y = ( n1y*(n1x + n1z) - n1x*n1x - n1z*n1z )/Local ;
        t2z = ( n1z*(n1x + n1y) - n1x*n1x - n1y*n1y )/Local ;
    }

    alpha2 = (t1x*nx + t1y*ny + t1z*nz) ; 
    if(alpha2 < 0) { t1x = -t1x ; t1y = -t1y ; t1z = -t1z ; alpha2 = -alpha2 ; }

    alpha3 = (t2x*nx + t2y*ny + t2z*nz) ; 
    if(alpha3 < 0) { t2x = -t2x ; t2y = -t2y ; t2z = -t2z ; alpha3 = -alpha3 ; }

    PetscReal smax1 = PDEHLLCRiemannSolver(VL, VR, n1x, n1y, n1z, x, y, z, Flux1);
    PetscReal smax2 = PDEHLLCRiemannSolver(VL, VR, t1x, t1y, t1z, x, y, z, Flux2);
    PetscReal smax3 = PDEHLLCRiemannSolver(VL, VR, t2x, t2y, t2z, x, y, z, Flux3);

    for(c = 0 ; c < 7; ++c) 
        Flux[c] = alpha1*Flux1[c] + alpha2*Flux2[c] + alpha3*Flux3[c];
    
    return PetscMax(smax1, PetscMax(smax2, smax3));
}

//----------------------------------------------------------------------------
// Viscous Riemann Solver (Does average of the two fluxes)  
//----------------------------------------------------------------------------

PetscReal PDEViscRiemannSolverPrim(const Field* VL, PetscReal grad_VL[nVar][DIM], 
                                   const Field* VR, PetscReal grad_VR[nVar][DIM], 
                                   const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                                   Field* Flux) {
    
    Field FL, FR; 
    PetscInt c; 
    
        
    PetscReal s_max_l = PDEViscFluxPrim(VL, grad_VL, nx, ny, nz, &FL); 
    PetscReal s_max_r = PDEViscFluxPrim(VR, grad_VR, nx, ny, nz, &FR);
    
    PetscReal s_max = PetscMax(s_max_l, s_max_r);
    
    for (c = 0; c < nVar; ++c) 
        Flux->comp[c] = 0.5*(FR.comp[c] + FL.comp[c]);
    
    return s_max; 
}
