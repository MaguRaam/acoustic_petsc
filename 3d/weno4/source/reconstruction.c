/*
 * reconstruction.c
 *      Author: sunder
 */ 
#include "hype.h"

//----------------------------------------------------------------------------
// Fourth order WENO reconstruction for cell averages  
//----------------------------------------------------------------------------


PetscReal evaluate_polynomial(const PetscReal x, const PetscReal y, const PetscReal z, const PetscReal coeffs[]) {
    return coeffs[0] + coeffs[1]*x + coeffs[2]*y + coeffs[3]*z + // Second order 
           coeffs[4]*(x*x - r1_12) + coeffs[5]*(y*y - r1_12) + coeffs[6]*(z*z - r1_12) + 
           coeffs[7]*x*y + coeffs[8]*y*z + coeffs[9]*z*x + // Third order 
           coeffs[10]*x*(x*x - r3_20) + coeffs[11]*y*(y*y - r3_20) + coeffs[12]*z*(z*z - r3_20) +
           coeffs[13]*(x*x - r1_12)*y + coeffs[14]*x*(y*y - r1_12) +  
           coeffs[15]*(y*y - r1_12)*z + coeffs[16]*y*(z*z - r1_12) + 
           coeffs[17]*(x*x - r1_12)*z + coeffs[18]*x*(z*z - r1_12) + 
           coeffs[19]*x*y*z;  // Fourth order 
}   


void evaluate_grad(const PetscReal coeffs[], PetscReal x, PetscReal y, PetscReal z, const PetscReal h, 
                                PetscReal* grad_x, PetscReal* grad_y, PetscReal* grad_z) {


    *grad_x = (coeffs[1] + 2.0*coeffs[4]*x + coeffs[7]*y + coeffs[9]*z + coeffs[10]*(3.0*x*x - r3_20) + 2.0*coeffs[13]*x*y + coeffs[14]*(y*y-r1_12) + 
               2.0*coeffs[17]*x*z + coeffs[18]*(z*z - r1_12) + coeffs[19]*y*z)/h;
    *grad_y = (coeffs[2] + 2.0*coeffs[5]*y + coeffs[7]*x + coeffs[8]*z + coeffs[11]*(3.0*y*y - r3_20) + coeffs[13]*(x*x - r1_12) + 2.0*coeffs[14]*x*y + 
               2.0*coeffs[15]*y*z + coeffs[16]*(z*z - r1_12) + coeffs[19]*x*z)/h; 
    *grad_z = (coeffs[3] + 2.0*coeffs[6]*z + coeffs[8]*y + coeffs[9]*x + coeffs[12]*(3.0*z*z - r3_20) + coeffs[15]*(y*y - r1_12) + 2.0*coeffs[16]*y*z + 
               coeffs[17]*(x*x - r1_12) + 2.0*coeffs[18]*x*z + coeffs[19]*x*y)/h;
}



void weno(const PetscReal U_x[], const PetscReal U_y[], const PetscReal U_z[], 
          const PetscReal U_xy[], const PetscReal U_yz[], const PetscReal U_zx[], const PetscReal U_xyz[], 
          PetscReal coeffs[])  {
    
    PetscReal gammaHi = 1.0; 
    PetscReal gammaLo = 0.0; 
    PetscReal p = 4.0;
    PetscReal total, wt_ratio; 
    PetscInt i; 
    
    PetscReal u_xR4,u_yR4,u_zR4,u_xxR4,u_yyR4,u_zzR4,u_xyR4,u_yzR4,u_xzR4,u_xxxR4,u_yyyR4,u_zzzR4,u_xxyR4,u_xyyR4,u_yyzR4,u_yzzR4,u_xxzR4,u_xzzR4,u_xyzR4;
    PetscReal w_R4, gamma_R4, IS_R4; 
    PetscReal u_xR3[7],u_yR3[7],u_zR3[7],u_xxR3[7],u_yyR3[7],u_zzR3[7],u_xyR3[7],u_yzR3[7],u_xzR3[7];
    PetscReal l_xR3,l_yR3,l_zR3,l_xxR3,l_yyR3,l_zzR3,l_xyR3,l_yzR3,l_xzR3;
    PetscReal nl_xR3,nl_yR3,nl_zR3,nl_xxR3,nl_yyR3,nl_zzR3,nl_xyR3,nl_yzR3,nl_xzR3;
    PetscReal gamma_R3[7], IS_R3[7], w_R3[7];
    
    gamma_R4 = gammaHi;
    gamma_R3[0] = (1.0 - gammaHi)*gammaLo; 
    
    for (i = 1 ; i < 7; ++i)
        gamma_R3[i] = r1_6*(1-gammaHi)*(1-gammaLo);
    
    
    // Cell average
    PetscReal u_0 = U_xy[0]; coeffs[0] = u_0; 
    
    // 1D Terms 
    PetscReal u_ip1 = U_x[3]; PetscReal u_jp1 = U_y[3]; PetscReal u_kp1 = U_z[3]; 
    PetscReal u_im1 = U_x[1]; PetscReal u_jm1 = U_y[1]; PetscReal u_km1 = U_z[1]; 
    PetscReal u_ip2 = U_x[4]; PetscReal u_jp2 = U_y[4]; PetscReal u_kp2 = U_z[4]; 
    PetscReal u_im2 = U_x[0]; PetscReal u_jm2 = U_y[0]; PetscReal u_km2 = U_z[0]; 
    
    // 2D Cross terms 
    PetscReal u_ip1jp1 = U_xy[1]; PetscReal u_jp1kp1 = U_yz[1]; PetscReal u_kp1ip1 = U_zx[1];
    PetscReal u_ip1jm1 = U_xy[2]; PetscReal u_jp1km1 = U_yz[2]; PetscReal u_kp1im1 = U_zx[2];
    PetscReal u_im1jp1 = U_xy[3]; PetscReal u_jm1kp1 = U_yz[3]; PetscReal u_km1ip1 = U_zx[3];
    PetscReal u_im1jm1 = U_xy[4]; PetscReal u_jm1km1 = U_yz[4]; PetscReal u_km1im1 = U_zx[4];
    
    // 3D Cross terms 
    
    PetscReal u_ip1jp1kp1  = U_xyz[0];
    PetscReal u_im1jp1kp1  = U_xyz[1];
    PetscReal u_ip1jm1kp1  = U_xyz[2];
    PetscReal u_ip1jp1km1  = U_xyz[3];
    PetscReal u_im1jm1kp1  = U_xyz[4];
    PetscReal u_im1jp1km1  = U_xyz[5];
    PetscReal u_ip1jm1km1  = U_xyz[6];
    PetscReal u_im1jm1km1  = U_xyz[7];
    
    // Fourth order stencil 
    
    u_xR4   = r1_120*(11.0*u_im2 - 82.0*u_im1 + 82.0*u_ip1 - 11.0*u_ip2);
    u_yR4   = r1_120*(11.0*u_jm2 - 82.0*u_jm1 + 82.0*u_jp1 - 11.0*u_jp2);
    u_zR4   = r1_120*(11.0*u_km2 - 82.0*u_km1 + 82.0*u_kp1 - 11.0*u_kp2);
    u_xxR4  = 0.5*(u_im1 -2.0*u_0 + u_ip1);
    u_yyR4  = 0.5*(u_jm1 -2.0*u_0 + u_jp1);
    u_zzR4  = 0.5*(u_km1 -2.0*u_0 + u_kp1);
    u_xyR4  = 0.25*(u_im1jm1 - u_im1jp1 - u_ip1jm1 + u_ip1jp1);
    u_yzR4  = 0.25*(u_jm1km1 - u_jm1kp1 - u_jp1km1 + u_jp1kp1);
    u_xzR4  = 0.25*(u_km1im1 - u_km1ip1 - u_kp1im1 + u_kp1ip1);
    u_xxxR4 = r1_12*(-u_im2 + 2.0*(u_im1 - u_ip1) + u_ip2);
    u_yyyR4 = r1_12*(-u_jm2 + 2.0*(u_jm1 - u_jp1) + u_jp2);
    u_zzzR4 = r1_12*(-u_km2 + 2.0*(u_km1 - u_kp1) + u_kp2);
    u_xxyR4 = 0.25*(-u_im1jm1 + u_im1jp1 - u_ip1jm1 + u_ip1jp1) + 0.5*(u_jm1 - u_jp1);
    u_xyyR4 = 0.5*(u_im1 - u_ip1) + 0.25*(u_ip1jp1 - u_im1jm1 - u_im1jp1  + u_ip1jm1);
    u_yyzR4 = 0.25*(-u_jm1km1 + u_jm1kp1 - u_jp1km1 + u_jp1kp1) + 0.5*(u_km1 - u_kp1);
    u_yzzR4 = 0.5*(u_jm1 - u_jp1) + 0.25*(u_jp1kp1 - u_jm1km1 - u_jm1kp1  + u_jp1km1); 
    u_xxzR4 = 0.25*(-u_km1im1 + u_kp1im1 - u_km1ip1 + u_kp1ip1) + 0.5*(u_km1 - u_kp1);
    u_xzzR4 = 0.5*(u_im1 - u_ip1) + 0.25*(u_kp1ip1 - u_km1im1 - u_kp1im1  + u_km1ip1);
    u_xyzR4 = 0.125*(u_ip1jp1kp1 - u_im1jp1kp1 - u_ip1jm1kp1 - u_ip1jp1km1 + u_im1jm1kp1 + u_im1jp1km1 + u_ip1jm1km1 - u_im1jm1km1);
    
    IS_R4 = u_xR4*u_xR4 + u_yR4*u_yR4 + u_zR4*u_zR4 +
            r13_3*(u_xxR4*u_xxR4 + u_yyR4*u_yyR4 + u_zzR4*u_zzR4) + 
            r7_6*(u_xyR4*u_xyR4 + u_yzR4*u_yzR4 + u_xzR4*u_xzR4) +
            0.2*(u_xR4*u_xxxR4 + u_yR4*u_yyyR4 + u_zR4*u_zzzR4) +
            4.7*(u_xxyR4*u_xxyR4 + u_xxzR4*u_xxzR4 + u_xyyR4*u_xyyR4 + u_xzzR4*u_xzzR4 + u_yyzR4*u_yyzR4 + u_yzzR4*u_yzzR4) +
            39.06*(u_xxxR4*u_xxxR4 + u_yyyR4*u_yyyR4 + u_zzzR4*u_zzzR4) + 
            r61_48*u_xyzR4*u_xyzR4; 
            
    w_R4 = gamma_R4/PetscPowReal((IS_R4 + small_num), p); 
    total = w_R4; 
    
    // Centered third order stencil 
    
    u_xR3[0]  = 0.5*(u_ip1 - u_im1);
    u_yR3[0]  = 0.5*(u_jp1 - u_jm1);
    u_zR3[0]  = 0.5*(u_kp1 - u_km1);
    u_xxR3[0] = 0.5*(u_im1 - 2.0*u_0 + u_ip1); 
    u_yyR3[0] = 0.5*(u_jm1 - 2.0*u_0 + u_jp1);
    u_zzR3[0] = 0.5*(u_km1 - 2.0*u_0 + u_kp1);
    u_xyR3[0] = 0.25*(u_ip1jp1 - u_ip1jm1 - u_im1jp1 + u_im1jm1); 
    u_yzR3[0] = 0.25*(u_jp1kp1 - u_jp1km1 - u_jm1kp1 + u_jm1km1);
    u_xzR3[0] = 0.25*(u_kp1ip1 - u_kp1im1 - u_km1ip1 + u_km1im1); 
    
    // +x sided third order stencil
    
    u_xR3[1]  = -1.5*u_0 + 2.0*u_ip1 - 0.5*u_ip2; 
    u_yR3[1]  = 0.5*(u_jp1 - u_jm1);
    u_zR3[1]  = 0.5*(u_kp1 - u_km1);
    u_xxR3[1] = 0.5*(u_0 - 2.0*u_ip1 + u_ip2); 
    u_yyR3[1] = 0.5*(u_jm1 - 2.0*u_0 + u_jp1);
    u_zzR3[1] = 0.5*(u_km1 - 2.0*u_0 + u_kp1);
    u_xyR3[1] = 0.5*(u_ip1jp1 -u_ip1jm1) - u_yR3[1];
    u_yzR3[1] = 0.25*(u_jp1kp1 - u_jp1km1 - u_jm1kp1 + u_jm1km1);
    u_xzR3[1] = 0.5*(u_kp1ip1 -u_km1ip1) - u_zR3[1];
    
    // -x sided stencil 
    
    u_xR3[2]  = -2.0*u_im1 + 0.5*u_im2 + 1.5*u_0;
    u_yR3[2]  = 0.5*(u_jp1 - u_jm1);
    u_zR3[2]  = 0.5*(u_kp1 - u_km1);
    u_xxR3[2] = 0.5*(u_im2 - 2.0*u_im1 + u_0); 
    u_yyR3[2] = 0.5*(u_jm1 - 2.0*u_0 + u_jp1);
    u_zzR3[2] = 0.5*(u_km1 - 2.0*u_0 + u_kp1);
    u_xyR3[2] = 0.5*(u_im1jm1 - u_im1jp1) + u_yR3[2]; 
    u_yzR3[2] = 0.25*(u_jp1kp1 - u_jp1km1 - u_jm1kp1 + u_jm1km1);
    u_xzR3[2] = 0.5*(u_km1im1 - u_kp1im1) + u_zR3[2];  
    
    // +y sided stencil 
    
    u_xR3[3]  = 0.5*(u_ip1 - u_im1);
    u_yR3[3]  = -1.5*u_0 + 2.0*u_jp1 - 0.5*u_jp2;
    u_zR3[3]  = 0.5*(u_kp1 - u_km1);
    u_xxR3[3] = 0.5*(u_im1 - 2.0*u_0 + u_ip1); 
    u_yyR3[3] = 0.5*(u_0 - 2.0*u_jp1 + u_jp2);
    u_zzR3[3] = 0.5*(u_km1 - 2.0*u_0 + u_kp1);
    u_xyR3[3] = 0.5*(u_ip1jp1 - u_im1jp1) - u_xR3[3];
    u_yzR3[3] = 0.5*(u_jp1kp1 - u_jp1km1) - u_zR3[3];
    u_xzR3[3] = 0.25*(u_kp1ip1 - u_kp1im1 - u_km1ip1 + u_km1im1); 
    
    // -y sided stencil 
    
    u_xR3[4]  = 0.5*(u_ip1 - u_im1);
    u_yR3[4]  = -2.0*u_jm1 + 0.5*u_jm2 + 1.5*u_0;
    u_zR3[4]  = 0.5*(u_kp1 - u_km1);
    u_xxR3[4] = 0.5*(u_im1 - 2.0*u_0 + u_ip1); 
    u_yyR3[4] = 0.5*(u_jm2 - 2.0*u_jm1 + u_0);
    u_zzR3[4] = 0.5*(u_km1 - 2.0*u_0 + u_kp1);
    u_xyR3[4] = 0.5*(u_im1jm1 - u_ip1jm1) + u_xR3[4];
    u_yzR3[4] = 0.5*(u_jm1km1 - u_jm1kp1) + u_zR3[4];
    u_xzR3[4] = 0.25*(u_kp1ip1 - u_kp1im1 - u_km1ip1 + u_km1im1); 
    
    // +z sided stencil 
    
    u_xR3[5]  = 0.5*(u_ip1 - u_im1);
    u_yR3[5]  = 0.5*(u_jp1 - u_jm1);
    u_zR3[5]  = -1.5*u_0 + 2.0*u_kp1 - 0.5*u_kp2;
    u_xxR3[5] = 0.5*(u_im1 - 2.0*u_0 + u_ip1); 
    u_yyR3[5] = 0.5*(u_jm1 - 2.0*u_0 + u_jp1);
    u_zzR3[5] = 0.5*(u_0 - 2.0*u_kp1 + u_kp2);
    u_xyR3[5] = 0.25*(u_ip1jp1 - u_ip1jm1 - u_im1jp1 + u_im1jm1); 
    u_yzR3[5] = 0.5*(u_jp1kp1 - u_jm1kp1) - u_yR3[5];
    u_xzR3[5] = 0.5*(u_kp1ip1 - u_kp1im1) - u_xR3[5];

    // -z sided stencil 
    
    u_xR3[6]  = 0.5*(u_ip1 - u_im1);
    u_yR3[6]  = 0.5*(u_jp1 - u_jm1);
    u_zR3[6]  = -2.0*u_km1 + 0.5*u_km2 + 1.5*u_0;
    u_xxR3[6] = 0.5*(u_im1 - 2.0*u_0 + u_ip1); 
    u_yyR3[6] = 0.5*(u_jm1 - 2.0*u_0 + u_jp1);
    u_zzR3[6] = 0.5*(u_km2 - 2.0*u_km1 + u_0);
    u_xyR3[6] = 0.25*(u_ip1jp1 - u_ip1jm1 - u_im1jp1 + u_im1jm1); 
    u_yzR3[6] = 0.5*(u_jm1km1 - u_jp1km1) + u_yR3[6]; 
    u_xzR3[6] = 0.5*(u_km1im1 - u_km1ip1) + u_xR3[6];
    
    for (i = 0; i < 7; ++i) {
        IS_R3[i] = (u_xR3[i]*u_xR3[i] + u_yR3[i]*u_yR3[i] + u_zR3[i]*u_zR3[i]) + 
                   r13_3*(u_xxR3[i]*u_xxR3[i] + u_yyR3[i]*u_yyR3[i] + u_zzR3[i]*u_zzR3[i]) + 
                    r7_6*(u_xyR3[i]*u_xyR3[i] + u_yzR3[i]*u_yzR3[i] + u_xzR3[i]*u_xzR3[i]) ;
                    
        w_R3[i] = gamma_R3[i]/PetscPowReal((IS_R3[i] + small_num), p);
        
        total += w_R3[i]; 
    }

    w_R4 = w_R4/total; 
    
    for (i = 0; i < 7; ++i) 
        w_R3[i] = w_R3[i]/total; 
    
    l_xR3=0.0;l_yR3=0.0;l_zR3=0.0;l_xxR3=0.0;l_yyR3=0.0;l_zzR3=0.0;l_xyR3=0.0;l_yzR3=0.0;l_xzR3=0.0;
    nl_xR3=0.0;nl_yR3=0.0;nl_zR3=0.0;nl_xxR3=0.0;nl_yyR3=0.0;nl_zzR3=0.0;nl_xyR3=0.0;nl_yzR3=0.0;nl_xzR3=0.0;
    
    for (i = 0; i < 7; ++i) {
                    
        l_xR3  += gamma_R3[i]*u_xR3[i];    nl_xR3  += w_R3[i]*u_xR3[i];
        l_yR3  += gamma_R3[i]*u_yR3[i];    nl_yR3  += w_R3[i]*u_yR3[i];
        l_zR3  += gamma_R3[i]*u_zR3[i];    nl_zR3  += w_R3[i]*u_zR3[i];
        l_xxR3 += gamma_R3[i]*u_xxR3[i];   nl_xxR3 += w_R3[i]*u_xxR3[i];
        l_yyR3 += gamma_R3[i]*u_yyR3[i];   nl_yyR3 += w_R3[i]*u_yyR3[i];
        l_zzR3 += gamma_R3[i]*u_zzR3[i];   nl_zzR3 += w_R3[i]*u_zzR3[i];
        l_xyR3 += gamma_R3[i]*u_xyR3[i];   nl_xyR3 += w_R3[i]*u_xyR3[i];
        l_yzR3 += gamma_R3[i]*u_yzR3[i];   nl_yzR3 += w_R3[i]*u_yzR3[i];
        l_xzR3 += gamma_R3[i]*u_xzR3[i];   nl_xzR3 += w_R3[i]*u_xzR3[i];
    }
    
    wt_ratio = w_R4/gamma_R4;
    
    coeffs[0]  = u_0; 
    coeffs[1]  = wt_ratio*(u_xR4 - l_xR3) + nl_xR3; 
    coeffs[2]  = wt_ratio*(u_yR4 - l_yR3) + nl_yR3; 
	coeffs[3]  = wt_ratio*(u_zR4 - l_zR3) + nl_zR3; 
    coeffs[4]  = wt_ratio*(u_xxR4 - l_xxR3) + nl_xxR3; 
    coeffs[5]  = wt_ratio*(u_yyR4 - l_yyR3) + nl_yyR3; 
    coeffs[6]  = wt_ratio*(u_zzR4 - l_zzR3) + nl_zzR3; 
    coeffs[7]  = wt_ratio*(u_xyR4 - l_xyR3) + nl_xyR3;
    coeffs[8]  = wt_ratio*(u_yzR4 - l_yzR3) + nl_yzR3; 
    coeffs[9]  = wt_ratio*(u_xzR4 - l_xzR3) + nl_xzR3; 
    coeffs[10] = wt_ratio*u_xxxR4;
    coeffs[11] = wt_ratio*u_yyyR4;
    coeffs[12] = wt_ratio*u_zzzR4;
    coeffs[13] = wt_ratio*u_xxyR4;
    coeffs[14] = wt_ratio*u_xyyR4;
    coeffs[15] = wt_ratio*u_yyzR4;
    coeffs[16] = wt_ratio*u_yzzR4;
    coeffs[17] = wt_ratio*u_xxzR4;
    coeffs[18] = wt_ratio*u_xzzR4;
    coeffs[19] = wt_ratio*u_xyzR4;
    
}
