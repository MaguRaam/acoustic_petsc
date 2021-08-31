/*
 * reconstruction.c
 *      Author: sunder
 */ 
#include "hype.h"

//----------------------------------------------------------------------------
// Third order WENO reconstruction for cell averages  
//----------------------------------------------------------------------------


PetscReal evaluate_polynomial(const PetscReal x, const PetscReal y, const PetscReal z, const PetscReal coeffs[]) {
    return coeffs[0] + coeffs[1]*x + coeffs[2]*y + coeffs[3]*z + 
           coeffs[4]*(x*x - r1_12) + coeffs[5]*(y*y - r1_12) + coeffs[6]*(z*z - r1_12) + 
           coeffs[7]*x*y + coeffs[8]*y*z + coeffs[9]*z*x; 
}

/*
void evaluate_grad(const PetscReal coeffs[], PetscReal x, PetscReal y, const PetscReal h, 
								PetscReal* grad_x, PetscReal* grad_y) {
	
	
	*grad_x = (coeffs[1] + 2.0*coeffs[3]*x + coeffs[5]*y + coeffs[6]*(3.0*x*x - r3_20) + 2.0*coeffs[8]*x*y + coeffs[9]*(y*y - r1_12))/h;
	*grad_y = (coeffs[2] + 2.0*coeffs[4]*y + coeffs[5]*x + coeffs[7]*(3.0*y*y - r3_20) + coeffs[8]*(x*x - r1_12) + 2.0*coeffs[9]*x*y)/h; 
}
*/


void weno(const PetscReal U_x[], const PetscReal U_y[], const PetscReal U_z[], 
          const PetscReal U_xy[], const PetscReal U_yz[], const PetscReal U_zx[],
          PetscReal coeffs[])  {
    
    const PetscReal central_cell_wt = 100.0;
    const PetscReal p = 4.0;
    PetscReal total, temp; 
    PetscInt i; 
    PetscReal u_x[3], u_xx[3], u_xy[4], IS[4], wt[4];
    
    // Cell average
    PetscReal u_0 = U_xy[0]; coeffs[0] = u_0; 
    
    // 1D Terms 
    PetscReal u_ip1 = U_x[3]; PetscReal u_jp1 = U_y[3]; PetscReal u_kp1 = U_z[3]; 
    PetscReal u_im1 = U_x[1]; PetscReal u_jm1 = U_y[1]; PetscReal u_km1 = U_z[1]; 
    PetscReal u_ip2 = U_x[4]; PetscReal u_jp2 = U_y[4]; PetscReal u_kp2 = U_z[4]; 
    PetscReal u_im2 = U_x[0]; PetscReal u_jm2 = U_y[0]; PetscReal u_km2 = U_z[0]; 
    
    // Cross terms 
    PetscReal u_ip1jp1 = U_xy[1]; PetscReal u_jp1kp1 = U_yz[1]; PetscReal u_kp1ip1 = U_zx[1];
    PetscReal u_ip1jm1 = U_xy[2]; PetscReal u_jp1km1 = U_yz[2]; PetscReal u_kp1im1 = U_zx[2];
    PetscReal u_im1jp1 = U_xy[3]; PetscReal u_jm1kp1 = U_yz[3]; PetscReal u_km1ip1 = U_zx[3];
    PetscReal u_im1jm1 = U_xy[4]; PetscReal u_jm1km1 = U_yz[4]; PetscReal u_km1im1 = U_zx[4];
    
    // Reconstruct in x-direction 

    u_x[0] = -2.0*u_im1 + 0.5*u_im2 + 1.5*u_0; u_xx[0] = 0.5*(u_im2 - 2.0*u_im1 + u_0);
    u_x[1] = 0.5*(u_ip1 - u_im1);              u_xx[1] = 0.5*(u_im1 - 2.0*u_0 + u_ip1);
    u_x[2] = -1.5*u_0 + 2.0*u_ip1 - 0.5*u_ip2; u_xx[2] = 0.5*(u_0 - 2.0*u_ip1 + u_ip2);
    
    for (i = 0; i < 3; ++i) {
        IS[i] = u_x[i]*u_x[i] + r13_3*u_xx[i]*u_xx[i];
        wt[i] = 1.0/PetscPowReal((IS[i] + small_num),p); 
    }
    
    wt[1] = central_cell_wt*wt[1];
    total = wt[0] + wt[1] + wt[2]; 
    wt[0] = wt[0]/total; wt[1] = wt[1]/total; wt[2] = wt[2]/total;  
    
    coeffs[1] = wt[0]*u_x[0] + wt[1]*u_x[1] + wt[2]*u_x[2]; 
    coeffs[4] = wt[0]*u_xx[0] + wt[1]*u_xx[1] + wt[2]*u_xx[2];
    
    // Reconstruct in y-direction 

    u_x[0] = -2.0*u_jm1 + 0.5*u_jm2 + 1.5*u_0; u_xx[0] = 0.5*(u_jm2 - 2.0*u_jm1 + u_0);
    u_x[1] = 0.5*(u_jp1 - u_jm1);              u_xx[1] = 0.5*(u_jm1 - 2.0*u_0 + u_jp1);
    u_x[2] = -1.5*u_0 + 2.0*u_jp1 - 0.5*u_jp2; u_xx[2] = 0.5*(u_0 - 2.0*u_jp1 + u_jp2);
    
    for (i = 0; i < 3; ++i) {
        IS[i] = u_x[i]*u_x[i] + r13_3*u_xx[i]*u_xx[i];
        wt[i] = 1.0/PetscPowReal((IS[i] + small_num),p); 
    }
    
    wt[1] = central_cell_wt*wt[1];
    total = wt[0] + wt[1] + wt[2]; 
    wt[0] = wt[0]/total; wt[1] = wt[1]/total; wt[2] = wt[2]/total;  
    
    coeffs[2] = wt[0]*u_x[0] + wt[1]*u_x[1] + wt[2]*u_x[2]; 
    coeffs[5] = wt[0]*u_xx[0] + wt[1]*u_xx[1] + wt[2]*u_xx[2]; 
    
    // Reconstruct in z-direction 

    u_x[0] = -2.0*u_km1 + 0.5*u_km2 + 1.5*u_0; u_xx[0] = 0.5*(u_km2 - 2.0*u_km1 + u_0);
    u_x[1] = 0.5*(u_kp1 - u_km1);              u_xx[1] = 0.5*(u_km1 - 2.0*u_0 + u_kp1);
    u_x[2] = -1.5*u_0 + 2.0*u_kp1 - 0.5*u_kp2; u_xx[2] = 0.5*(u_0 - 2.0*u_kp1 + u_kp2);
    
    for (i = 0; i < 3; ++i) {
        IS[i] = u_x[i]*u_x[i] + r13_3*u_xx[i]*u_xx[i];
        wt[i] = 1.0/PetscPowReal((IS[i] + small_num),p); 
    }
    
    wt[1] = central_cell_wt*wt[1];
    total = wt[0] + wt[1] + wt[2]; 
    wt[0] = wt[0]/total; wt[1] = wt[1]/total; wt[2] = wt[2]/total;  
    
    coeffs[3] = wt[0]*u_x[0] + wt[1]*u_x[1] + wt[2]*u_x[2]; 
    coeffs[6] = wt[0]*u_xx[0] + wt[1]*u_xx[1] + wt[2]*u_xx[2]; 
    
    // Reconstruction in xy-direction
    
    u_xy[0] =  u_ip1jp1 - u_0 - coeffs[1] - coeffs[2] - coeffs[4] - coeffs[5];
    u_xy[1] = -u_ip1jm1 + u_0 + coeffs[1] - coeffs[2] + coeffs[4] + coeffs[5];
    u_xy[2] = -u_im1jp1 + u_0 - coeffs[1] + coeffs[2] + coeffs[4] + coeffs[5];
    u_xy[3] =  u_im1jm1 - u_0 + coeffs[1] + coeffs[2] - coeffs[4] - coeffs[5];
    
    temp = 4.0*(coeffs[4]*coeffs[4] + coeffs[5]*coeffs[5]);
    
    for (i = 0; i < 4; ++i) {
        IS[i] = temp + u_xy[i]*u_xy[i]; 
        wt[i] = 1.0/PetscPowReal((IS[i] + small_num),p); 
    }
    
    total = wt[0] + wt[1] + wt[2] + wt[3]; 
    wt[0] = wt[0]/total; wt[1] = wt[1]/total; wt[2] = wt[2]/total; wt[3] = wt[3]/total; 
    
    coeffs[7] = wt[0]*u_xy[0] + wt[1]*u_xy[1] + wt[2]*u_xy[2] + wt[3]*u_xy[3];
              
    // Reconstruction in yz-direction
    
    u_xy[0] =  u_jp1kp1 - u_0 - coeffs[2] - coeffs[3] - coeffs[5] - coeffs[6];
    u_xy[1] = -u_jp1km1 + u_0 + coeffs[2] - coeffs[3] + coeffs[5] + coeffs[6];
    u_xy[2] = -u_jm1kp1 + u_0 - coeffs[2] + coeffs[3] + coeffs[5] + coeffs[6];
    u_xy[3] =  u_jm1km1 - u_0 + coeffs[2] + coeffs[3] - coeffs[5] - coeffs[6];
    
    temp = 4.0*(coeffs[5]*coeffs[5] + coeffs[6]*coeffs[6]);
    
    for (i = 0; i < 4; ++i) {
        IS[i] = temp + u_xy[i]*u_xy[i]; 
        wt[i] = 1.0/PetscPowReal((IS[i] + small_num),p); 
    }
    
    total = wt[0] + wt[1] + wt[2] + wt[3]; 
    wt[0] = wt[0]/total; wt[1] = wt[1]/total; wt[2] = wt[2]/total; wt[3] = wt[3]/total; 
    
    coeffs[8] = wt[0]*u_xy[0] + wt[1]*u_xy[1] + wt[2]*u_xy[2] + wt[3]*u_xy[3];
    
    // Reconstruction in zx-direction
    
    u_xy[0] =  u_kp1ip1 - u_0 - coeffs[3] - coeffs[1] - coeffs[6] - coeffs[4];
    u_xy[1] = -u_kp1im1 + u_0 + coeffs[3] - coeffs[1] + coeffs[6] + coeffs[4];
    u_xy[2] = -u_km1ip1 + u_0 - coeffs[3] + coeffs[1] + coeffs[6] + coeffs[4];
    u_xy[3] =  u_km1im1 - u_0 + coeffs[3] + coeffs[1] - coeffs[6] - coeffs[4];
    
    temp = 4.0*(coeffs[6]*coeffs[6] + coeffs[4]*coeffs[4]);
    
    for (i = 0; i < 4; ++i) {
        IS[i] = temp + u_xy[i]*u_xy[i]; 
        wt[i] = 1.0/PetscPowReal((IS[i] + small_num),p); 
    }
    
    total = wt[0] + wt[1] + wt[2] + wt[3]; 
    wt[0] = wt[0]/total; wt[1] = wt[1]/total; wt[2] = wt[2]/total; wt[3] = wt[3]/total; 
    
    coeffs[9] = wt[0]*u_xy[0] + wt[1]*u_xy[1] + wt[2]*u_xy[2] + wt[3]*u_xy[3];
    
    /*
    coeffs[1] = 0.5*(u_ip1 - u_im1); 
    coeffs[2] = 0.0;  
    coeffs[3] = 0.0;   
    coeffs[4] = 0.5*(u_im1 - 2.0*u_0 + u_ip1);
    coeffs[5] = 0.0; 
    coeffs[6] = 0.0; 
    coeffs[7] = 0.0;
    coeffs[8] = 0.0; 
    coeffs[9] = 0.0;  
    */
              
}
