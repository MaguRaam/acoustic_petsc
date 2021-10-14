#include "hype.h"

/* Definitions of various PDE functions */

//----------------------------------------------------------------------------
// Convert a conserved variable to primitive variable
//----------------------------------------------------------------------------

void PDECons2Prim(const Field* Q, Field* V) {
    
    PetscReal r1_rho = 1.0/Q->comp[0]; 
    
    V->comp[0] = Q->comp[0];
    V->comp[1] = r1_rho*Q->comp[1];
    V->comp[2] = r1_rho*Q->comp[2];
    V->comp[3] = r1_rho*Q->comp[3];
    V->comp[4] = (GAMMA -1.0)*( Q->comp[4] - 0.5*r1_rho*( (Q->comp[1]*Q->comp[1] + Q->comp[2]*Q->comp[2] + Q->comp[3]*Q->comp[3]) ) );
}

//----------------------------------------------------------------------------
// Convert a primitive variable to conserved variable
//----------------------------------------------------------------------------

void PDEPrim2Cons(const Field* V, Field* Q) {
    
    PetscReal e = V->comp[4]/(GAMMA - 1.0);
    PetscReal k = 0.5*V->comp[0]*(V->comp[1]*V->comp[1] + V->comp[2]*V->comp[2] + V->comp[3]*V->comp[3]);

    Q->comp[0] = V->comp[0];
    Q->comp[1] = V->comp[0]*V->comp[1];
    Q->comp[2] = V->comp[0]*V->comp[2];
    Q->comp[3] = V->comp[0]*V->comp[3]; 
    Q->comp[4] = k + e;
}

//----------------------------------------------------------------------------
// Find the conservative flux components F in the given normal direction
//----------------------------------------------------------------------------

PetscReal PDEConsFlux (const Field* Q, 
                       const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                       const PetscReal x,  const PetscReal y,  const PetscReal z, 
                       Field* F) {


    PetscReal rho = Q->comp[0];
    PetscReal r1_rho = 1.0/rho; 
    PetscReal u = r1_rho*Q->comp[1];
    PetscReal v = r1_rho*Q->comp[2];
    PetscReal w = r1_rho*Q->comp[3];
    PetscReal E = Q->comp[4]; 
    PetscReal p = (GAMMA -1.0)*( E - 0.5*rho*(u*u + v*v + w*w) );

    // Check if the input state is physically admissible 

    if (rho < rho_floor) {
	    printf("Negative density = %f\n", rho);
	    printf("At x = %f, y = %f, z = %f\n", x, y, z);  
	    MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    if (p < prs_floor) {
	    printf("Negative pressure = %f\n", p);
	    printf("At x = %f, y = %f, z = %f\n", x, y, z);  
	  	MPI_Abort(PETSC_COMM_WORLD, 1);
    }

    // Now find the fluxes 

    F->comp[0] = nx*rho*u         + ny*rho*v         + nz*rho*w;
    F->comp[1] = nx*(rho*u*u + p) + ny*rho*u*v       + nz*rho*u*w;
    F->comp[2] = nx*rho*u*v       + ny*(rho*v*v + p) + nz*rho*v*w;
    F->comp[3] = nx*rho*u*w       + ny*rho*v*w       + nz*(rho*w*w + p);
    F->comp[4] = (E + p)*(nx*u + ny*v + nz*w);

    PetscReal a = PetscSqrtReal(r1_rho*GAMMA*p); 

    // Also obtain the maximum eigen value 

    PetscReal s_max = PetscAbsReal(u*nx + v*ny + w*nz) + a;

    return s_max;
}

//----------------------------------------------------------------------------
// Viscous Flux 
//----------------------------------------------------------------------------


PetscReal PDEViscFlux(const Field* Q, PetscReal grad_Q[nVar][DIM], PetscReal nx, PetscReal ny, PetscReal nz, Field* F) {
    

    PetscReal rho, rho2, irho, u, v, w, p, mu, k, rho_x, rhou_x, rhov_x, rhow_x, e_x, rho_y, rhou_y, rhov_y, rhow_y, e_y, rho_z, rhou_z, rhov_z, rhow_z, e_z, e, T;
    PetscReal rhou, rhov, rhow; 
    PetscReal u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z, T_x, T_y, T_z, div_v;
    PetscReal tau_xx, tau_xy, tau_xz, tau_yz, tau_yy, tau_zz, q_x, q_y, q_z;
    
    rho = Q->comp[0]; rho2 = Q->comp[0]*Q->comp[0]; irho = 1.0/Q->comp[0];  
    u = irho*Q->comp[1]; 
    v = irho*Q->comp[2];
    w = irho*Q->comp[3];
    rhou = Q->comp[1]; 
    rhov = Q->comp[2];
    rhow = Q->comp[3];
    e = Q->comp[4];
    p = (GAMMA -1.0)*( e - 0.5*rho*(u*u + v*v + w*w) );
    T = p/rho;
    
    mu = MU_0;
    k = C_V*mu*GAMMA/PR;
    
    
    rho_x  = grad_Q[0][0]; rho_y  = grad_Q[0][1]; rho_z  = grad_Q[0][2];
    rhou_x = grad_Q[1][0]; rhou_y = grad_Q[1][1]; rhou_z = grad_Q[1][2];
    rhov_x = grad_Q[2][0]; rhov_y = grad_Q[2][1]; rhov_z = grad_Q[2][2];
    rhow_x = grad_Q[3][0]; rhow_y = grad_Q[3][1]; rhow_z = grad_Q[3][2];
    e_x    = grad_Q[4][0];    e_y = grad_Q[4][1];    e_z = grad_Q[4][2];
    
    u_x = rhou_x - u*rho_x/rho;
    v_x = rhov_x - v*rho_x/rho;
    w_x = rhow_x - w*rho_x/rho;
    T_x = (-(GAMMA - 1.0)*rho_x/rho2) * (e -0.5*(rhou*rhou + rhov*rhov + rhow*rhow)/rho)
            + ((GAMMA - 1.0)/rho)*(e_x -(rhou*rhou_x + rhov*rhov_x + rhow*rhow_x)/rho + 0.5*(rhou*rhou + rhov*rhov + rhow*rhow)*rho_x/rho2);

    u_y = rhou_y - u*rho_y/rho;
    v_y = rhov_y - v*rho_y/rho;
    w_y = rhow_y - w*rho_y/rho;
    T_y = (-(GAMMA - 1.0)*rho_y/rho2) * (e -0.5*(rhou*rhou + rhov*rhov + rhow*rhow)/rho)
            + ((GAMMA - 1.0)/rho)*(e_y -(rhou*rhou_y + rhov*rhov_y + rhow*rhow_y)/rho + 0.5*(rhou*rhou + rhov*rhov + rhow*rhow)*rho_y/rho2);

    u_z = rhou_z - u*rho_z/rho;
    v_z = rhov_z - v*rho_z/rho;
    w_z = rhow_z - w*rho_z/rho;
    T_z = (-(GAMMA - 1.0)*rho_z/rho2) * (e -0.5*(rhou*rhou + rhov*rhov + rhow*rhow)/rho)
            + ((GAMMA - 1.0)/rho)*(e_z -(rhou*rhou_z + rhov*rhov_z + rhow*rhow_z)/rho + 0.5*(rhou*rhou + rhov*rhov + rhow*rhow)*rho_z/rho2);
    
    div_v = -(2.0/3.0)*mu*(u_x + v_y + w_z);
          
    tau_xx = div_v + 2.0*mu*u_x;
    tau_xy = mu*(u_y + v_x) ;
    tau_xz = mu*(u_z + w_x) ;
    tau_yy = div_v + 2.0*mu*v_y;
    tau_yz = mu*(v_z + w_y) ;
    tau_zz = div_v + 2.0*mu*w_z;

    q_x = -k*T_x;
    q_y = -k*T_y;
    q_z = -k*T_z;
    
    F->comp[0] = 0.0;
    F->comp[1] = -(nx*tau_xx + ny*tau_xy + nz*tau_xz);
    F->comp[2] = -(nx*tau_xy + ny*tau_yy + nz*tau_yz);
    F->comp[3] = -(nx*tau_xz + ny*tau_yz + nz*tau_zz);
    F->comp[4] = -(nx*(u*tau_xx + v*tau_xy + w*tau_xz - q_x) + ny*(u*tau_xy + v*tau_yy + w*tau_yz - q_y) + nz*(u*tau_xz + v*tau_yz + w*tau_zz - q_z));

    
    return PetscMax(r4_3*mu/Q->comp[0], GAMMA*mu/(PR*Q->comp[0])); 
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

PetscReal PDEHLLCRiemannSolver(const Field* QL, const Field* QR, 
                               const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                               const PetscReal x, const PetscReal y,  const PetscReal z, 
                               Field* Flux) {
        
    Field FL, FR; PetscInt c; 

    PetscReal s_max_l = PDEConsFlux(QL, nx, ny, nz, x, y, z, &FL); 
    PetscReal s_max_r = PDEConsFlux(QR, nx, ny, nz, x, y, z, &FR);

    PetscReal s_max = PetscMax(s_max_l, s_max_r);
    
    PetscReal rho_L, rho_R, u_L, u_R, v_L, v_R, w_L, w_R, P_L, P_R, c_L, c_R, E_L, E_R ;
    PetscReal un_L, un_R, ut1_L, ut1_R, ut2_L, ut2_R ;
    PetscReal t1x, t1y, t1z, t2x, t2y, t2z ;
    PetscReal un, ut1, ut2 ;
    PetscReal S_L, S_R, S_star ;
    Field VL, VR, QL_star,QR_star, FL_star, FR_star;
    PetscReal Rot_Mat[3][3], InvRot_Mat[3][3], Local;

    // Extract pimitive variables 
    
    PDECons2Prim(QL, &VL);
    PDECons2Prim(QR, &VR);
    
    rho_L = VL.comp[0]; u_L = VL.comp[1]; v_L = VL.comp[2]; w_L = VL.comp[3]; P_L = VL.comp[4];
    rho_R = VR.comp[0]; u_R = VR.comp[1]; v_R = VR.comp[2]; w_R = VR.comp[3]; P_R = VR.comp[4];
                
    // Define the two tangent directions following Miller and Colella's JCP paper
    
    if( PetscAbsReal(ny + nz) >  PetscAbsReal(ny - nz) ) {
        Local = PetscSqrtReal(2.0*(1.0 + nz*(ny - nx) + nx*ny) );
        t1x = (ny + nz)/Local ; t1y = (nz - nx)/Local ; t1z = -(nx + ny)/Local;
        t2x = ( nx*(nz - ny) - ny*ny - nz*nz )/Local; 
        t2y = ( ny*(nx + nz) + nx*nx + nz*nz )/Local; 
        t2z = ( nz*(nx - nz) - nx*nx - ny*ny )/Local; 
    } 
    
    else {
        Local = sqrt( 2.0*(1.0 - nx*ny - nx*nz - ny*nz) ) ;
        t1x = (ny - nz)/Local ; t1y = (nz - nx)/Local ; t1z = (nx - ny)/Local ;
        t2x = ( nx*(ny + nz) - ny*ny - nz*nz )/Local ; 
        t2y = ( ny*(nx + nz) - nx*nx - nz*nz )/Local ;
        t2z = ( nz*(nx + ny) - nx*nx - ny*ny )/Local ;
    }
    
    Rot_Mat[0][0] = nx;  Rot_Mat[0][1] = ny;  Rot_Mat[0][2] = nz ;
    Rot_Mat[1][0] = t1x; Rot_Mat[1][1] = t1y; Rot_Mat[1][2] = t1z ;
    Rot_Mat[2][0] = t2x; Rot_Mat[2][1] = t2y; Rot_Mat[2][2] = t2z ;

    invert3x3matrix(Rot_Mat, InvRot_Mat);

    un_L = u_L*nx + v_L*ny + w_L*nz ; ut1_L = u_L*t1x + v_L*t1y + w_L*t1z ; ut2_L = u_L*t2x + v_L*t2y + w_L*t2z ;
    un_R = u_R*nx + v_R*ny + w_R*nz ; ut1_R = u_R*t1x + v_R*t1y + w_R*t1z ; ut2_R = u_R*t2x + v_R*t2y + w_R*t2z ;

    c_L = PetscSqrtReal(GAMMA*P_L/rho_L); c_R = PetscSqrtReal(GAMMA*P_R/rho_R);

    E_L = QL->comp[4];
    E_R = QR->comp[4];

    S_L = PetscMin((un_R - c_R), (un_L - c_L)) ;
    S_R = PetscMax((un_L + c_L), (un_R + c_R)) ;
    S_star = ( P_R - P_L + rho_L*un_L*(S_L-un_L) - rho_R*un_R*(S_R - un_R) )/(rho_L*(S_L - un_L) - rho_R*(S_R - un_R)) ;

    // Find the star state 
    
    QL_star.comp[0] = rho_L*(S_L - un_L)/(S_L - S_star)  ;
    un = S_star; ut1 = ut1_L; ut2 = ut2_L;
    QL_star.comp[1] = QL_star.comp[0]*(un*InvRot_Mat[0][0] + ut1*InvRot_Mat[0][1] + ut2*InvRot_Mat[0][2]);
    QL_star.comp[2] = QL_star.comp[0]*(un*InvRot_Mat[1][0] + ut1*InvRot_Mat[1][1] + ut2*InvRot_Mat[1][2]);
    QL_star.comp[3] = QL_star.comp[0]*(un*InvRot_Mat[2][0] + ut1*InvRot_Mat[2][1] + ut2*InvRot_Mat[2][2]);
    QL_star.comp[4] = QL_star.comp[0]*( (E_L/rho_L) + (S_star - un_L)*(S_star + P_L/(rho_L*(S_L - un_L)) ) );

    QR_star.comp[0] = rho_R*(S_R - un_R)/(S_R - S_star) ;
    un = S_star ; ut1 = ut1_R ; ut2 = ut2_R ;
    QR_star.comp[1] = QR_star.comp[0]*(un*InvRot_Mat[0][0] + ut1*InvRot_Mat[0][1] + ut2*InvRot_Mat[0][2]);
    QR_star.comp[2] = QR_star.comp[0]*(un*InvRot_Mat[1][0] + ut1*InvRot_Mat[1][1] + ut2*InvRot_Mat[1][2]);
    QR_star.comp[3] = QR_star.comp[0]*(un*InvRot_Mat[2][0] + ut1*InvRot_Mat[2][1] + ut2*InvRot_Mat[2][2]);
    QR_star.comp[4] = QR_star.comp[0]*( (E_R/rho_R) + (S_star - un_R)*(S_star + P_R/(rho_R*(S_R - un_R)) ) );

    for(c = 0 ; c < nVar; ++c) {
        FL_star.comp[c] = FL.comp[c] + S_L*(QL_star.comp[c] - QL->comp[c]) ; 
    }

    for(c = 0 ; c < nVar; ++c) {
        FR_star.comp[c] = FR.comp[c] + S_R*(QR_star.comp[c] - QR->comp[c]);
    } 
    
    // Based on wave speed select the appropriate flux 
    
    if( S_L > 0.0 ) {
        for(c = 0 ; c < nVar; ++c) 
            Flux->comp[c] = FL.comp[c];
    } 
    
    else if((S_star >= 0.0) && (S_L < 0.0)) {
        for(c = 0 ; c < nVar; ++c) 
            Flux->comp[c] = FL_star.comp[c];
    } 
    
    else if((S_star < 0.0) && (S_R >= 0.0)) {
        for(c = 0 ; c < nVar; ++c) 
            Flux->comp[c] = FR_star.comp[c];
    } 
    
    else if(S_R < 0.0) {
        for(c = 0 ; c < nVar; ++c) 
            Flux->comp[c] = FR.comp[c];
    }

    return s_max; 
}

//----------------------------------------------------------------------------
// Rotated HLLC Riemann Solver (adds diffusion along the tangential direction)
//----------------------------------------------------------------------------

PetscReal PDErotHLLCRiemannSolver(const Field* QL, const Field* QR, 
                               const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                               const PetscReal x, const PetscReal y,  const PetscReal z, 
                               Field* Flux) {
    
    Field Flux1, Flux2, Flux3; 

    PetscInt c;
    PetscReal alpha1, alpha2, alpha3, n1x, n1y, n1z, t1x, t1y, t1z, t2x, t2y, t2z, u_L, u_R, v_L, v_R, w_L, w_R, du, dv, dw, dq, Local, irho ;

    irho = 1.0/QL->comp[0]; 
    u_L = irho*QL->comp[1]; v_L = irho*QL->comp[2]; w_L = irho*QL->comp[3];
    
    irho = 1.0/QR->comp[0];
    u_R = irho*QR->comp[1]; v_R = irho*QR->comp[2]; w_R = irho*QR->comp[3];
    
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
        t2z = ( n1z*(n1x - n1y) - n1x*n1x - n1y*n1y )/Local ; // Wrong here. Now corrected 
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

    PetscReal smax1 = PDEHLLCRiemannSolver(QL, QR, n1x, n1y, n1z, x, y, z, &Flux1);
    PetscReal smax2 = PDEHLLCRiemannSolver(QL, QR, t1x, t1y, t1z, x, y, z, &Flux2);
    PetscReal smax3 = PDEHLLCRiemannSolver(QL, QR, t2x, t2y, t2z, x, y, z, &Flux3);

    for(c = 0 ; c < nVar; ++c) 
        Flux->comp[c] = alpha1*Flux1.comp[c] + alpha2*Flux2.comp[c] + alpha3*Flux3.comp[c];
    
    return PetscMax(smax1, PetscMax(smax2, smax3));
}

//----------------------------------------------------------------------------
// Viscous Riemann Solver (Does average of the two fluxes)  
//----------------------------------------------------------------------------

PetscReal PDEViscRiemannSolver(const Field* QL, PetscReal grad_QL[nVar][DIM], 
                                   const Field* QR, PetscReal grad_QR[nVar][DIM], 
                                   const PetscReal nx, const PetscReal ny, const PetscReal nz, 
                                   Field* Flux) {
    
    Field FL, FR; 
    PetscInt c; 
    
        
    PetscReal s_max_l = PDEViscFlux(QL, grad_QL, nx, ny, nz, &FL); 
    PetscReal s_max_r = PDEViscFlux(QR, grad_QR, nx, ny, nz, &FR);
    
    PetscReal s_max = PetscMax(s_max_l, s_max_r);
    
    for (c = 0; c < nVar; ++c) 
        Flux->comp[c] = 0.5*(FR.comp[c] + FL.comp[c]);
    
    return s_max; 
}

