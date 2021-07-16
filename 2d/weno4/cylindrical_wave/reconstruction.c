/*
 * reconstruction.c
 *      Author: sunder
 */ 
#include "hype.h"

//----------------------------------------------------------------------------
// Value of Nth Order basis functions 
//----------------------------------------------------------------------------

PetscReal basis(PetscReal x, PetscReal y, PetscInt n) {
    
    switch (n) {
        case 0:
            return 1.0;
            break; 
        case 1:
            return x; 
            break;
        case 2:
            return y;
            break;
        case 3:
            return x*x - 1./12.;
            break;
        case 4:
            return y*y - 1./12.;
            break;
        case 5:
            return x*y;
            break; 
        case 6:
            return x*(x*x - 3./20.);
            break;
        case 7:
            return y*(y*y - 3./20.);
            break; 
        case 8:
            return y*(x*x - 1./12.);
            break; 
        case 9:
            return x*(y*y - 1./12.);
            break;
        default:
            return 0.0;
    }
}

//----------------------------------------------------------------------------
// Gradients of Nth Order basis functions 
//----------------------------------------------------------------------------

void basis_grad(PetscReal x, PetscReal y, PetscInt n, PetscReal* grad_x, PetscReal* grad_y) {
    
    switch (n) {
        case 0:
            *grad_x = 0.0;
            *grad_y = 0.0; 
            break; 
        case 1:
            *grad_x = 1.0;
            *grad_y = 0.0; 
            break;
        case 2:
            *grad_x = 0.0;
            *grad_y = 1.0; 
            break;
        case 3:
            *grad_x = 2.0*x;
            *grad_y = 0.0; 
            break;
        case 4:
            *grad_x = 0.0;
            *grad_y = 2.0*y; 
            break;
        case 5:
            *grad_x = y;
            *grad_y = x; 
            break; 
        case 6:
            *grad_x = 3.0*x*x - 3./20.;
            *grad_y = 0.0; ;
            break;
        case 7:
            *grad_x = 0.0;
            *grad_y = 3.0*y*y - 3./20.;
            break; 
        case 8:
            *grad_x = 2.0*x*y;
            *grad_y = (x*x - 1./12.); 
            break; 
        case 9:
            *grad_x = (y*y - 1./12.);
            *grad_y = 2.0*x*y;
            break;
        default:
            *grad_x = 0.0;
            *grad_y = 0.0; 
    }
}

//----------------------------------------------------------------------------
// 2D 4th order WENO reconstruction 
//----------------------------------------------------------------------------

void weno(const PetscReal U_x[], const PetscReal U_y[], const PetscReal U_xy[], PetscReal coeffs[]) {
    
    PetscReal u_0 = U_xy[0]; 
    PetscReal u_ip1 = U_x[3]; PetscReal u_jp1 = U_y[3]; PetscReal u_ip1jp1 = U_xy[1];
    PetscReal u_im1 = U_x[1]; PetscReal u_jm1 = U_y[1]; PetscReal u_ip1jm1 = U_xy[2];
    PetscReal u_ip2 = U_x[4]; PetscReal u_jp2 = U_y[4]; PetscReal u_im1jp1 = U_xy[3];
    PetscReal u_im2 = U_x[0]; PetscReal u_jm2 = U_y[0]; PetscReal u_im1jm1 = U_xy[4];

    coeffs[0] = u_0;
    coeffs[1] = r41_60*(-u_im1 + u_ip1) + r11_120*(u_im2 - u_ip2);
    coeffs[2] = r41_60*(-u_jm1 + u_jp1) + r11_120*(u_jm2 - u_jp2);
    coeffs[3] = -u_0 + 0.5*(u_im1 + u_ip1);
    coeffs[4] = -u_0 + 0.5*(u_jm1 + u_jp1);
    coeffs[5] = 0.25*(u_im1jm1 - u_im1jp1 - u_ip1jm1 + u_ip1jp1);
    coeffs[6] = r1_6*(u_im1 - u_ip1) + r1_12*(u_ip2 - u_im2);
    coeffs[7] = r1_6*(u_jm1 - u_jp1) + r1_12*(u_jp2 - u_jm2);
    coeffs[8] = 0.25*(-u_im1jm1 + u_im1jp1 - u_ip1jm1 + u_ip1jp1) + 0.5*(u_jm1 - u_jp1);
    coeffs[9] = 0.5*(u_im1 - u_ip1) + 0.25*(u_ip1jp1 - u_im1jm1 - u_im1jp1  + u_ip1jm1);

}
