/*
 * initial_condition.c
 *      Author: sunder
 */ 
#include "hype.h" 

//----------------------------------------------------------------------------
// Initial condition function 
//----------------------------------------------------------------------------

Field InitialCondition(PetscReal x, PetscReal y, PetscReal z) {

    Field Q0;
    Field V0;

    //----------------------------------------------------------------------- 
    
    if (x > 2.8314606741573036) {

                V0.comp[0] = 1.92691;
                V0.comp[1] = -0.3336067590079454;
                V0.comp[2] = 0.0;
                V0.comp[3] = 0.0;
                V0.comp[4] = 1.5698;
                V0.comp[5] = 1.0;
    }

    else  {

                V0.comp[0] = 1.4;
                V0.comp[1] = 0.0;
                V0.comp[2] = 0.0;
                V0.comp[3] = 0.0;
                V0.comp[4] = 1.0;
                V0.comp[5] = 1.0;
    }

    // Helium

    PetscReal x_0 = 2.05247191011236; PetscReal y_0 = 0.5; PetscReal z_0 = 0.5;
    PetscReal r = 0.2808988764044944;

    if ((x-x_0)*(x-x_0) + (y-y_0)*(y-y_0) + (z-z_0)*(z-z_0) <= r*r ) {

            V0.comp[0] = 0.2546;
            V0.comp[1] = 0.0;
            V0.comp[2] = 0.0;
            V0.comp[3] = 0.0;
            V0.comp[4] = 1.0;
            V0.comp[5] = 0.0;

    }

	/*
	PetscReal M_s = 3.0; 
	PetscReal rho_1 = 1.4; 
	PetscReal p_1 = 1.0; 
	PetscReal a_1 = PetscSqrtReal(g2*p_1/rho_1);   
	PetscReal prs_ratio = 1.0 + (2.0*g2)/(g2 + 1.0)*(M_s*M_s - 1.0); 
	PetscReal tempa = (g2 + 1.0)/(g2 - 1.0);
	PetscReal rho_ratio = (1.0 + tempa*prs_ratio)/(tempa + prs_ratio);
	tempa = (g2-1.0)/(g2+1.0);
    PetscReal tempb = ((2.0*g2)/(g2+1.0))/(prs_ratio + tempa);
    PetscReal u_p = (a_1/g2)*(prs_ratio -1.0)*PetscSqrtReal(tempb); 

    PetscReal h = 0.05;
    PetscReal smear = 2.0; 

    PetscReal rhoL = rho_1*rho_ratio;
    PetscReal rhoR = rho_1; 

    PetscReal pL = p_1*prs_ratio;
    PetscReal pR = p_1; 

    PetscReal uL = u_p; 
    PetscReal uR = 0.0;

    PetscReal rhoW = 1000.0;
    PetscReal phiW = 1.0;
    PetscReal phiA = 0.0; 

	PetscReal x0 = 2.0; PetscReal y0 = 0.0; PetscReal z0 = 0.0;
	PetscReal R0 = 0.5; 
    PetscReal R = PetscSqrtReal((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0)); 

    V0.comp[0] = 0.5*((rhoL+rhoR) + (rhoR-rhoL)*PetscTanhReal((x-1.0)/(smear*h)));
    V0.comp[1] = 0.5*((uL+uR) + (uR-uL)*PetscTanhReal((x-1.0)/(smear*h)));
    V0.comp[2] = 0.0;
    V0.comp[3] = 0.0;
    V0.comp[4] = 0.5*((pL+pR) + (pR-pL)*PetscTanhReal((x-1.0)/(smear*h)));
    V0.comp[5] = 0.5*((phiW+phiA) + (phiA-phiW)*PetscTanhReal((R-R0)/(smear*h)));	

    V0.comp[0] = V0.comp[0] + (rhoW-rho_1)*V0.comp[5]; 
    */
    /*
	// Air part 
	
	if (x < 1.0) {
		
		// Air
		
		V0.comp[0] = rho_1*rho_ratio;
		V0.comp[1] = u_p;
		V0.comp[2] = 0.0;
        V0.comp[3] = 0.0; 
		V0.comp[4] = p_1*prs_ratio;
		V0.comp[5] = 0.01; 
	}
	
	else {
		
		V0.comp[0] = rho_1;
		V0.comp[1] = 0.0;
		V0.comp[2] = 0.0; 
        V0.comp[3] = 0.0;
		V0.comp[4] = p_1;
		V0.comp[5] = 0.01; 
	}

    */
	
	// Water
	/*
	PetscReal x_0 = 2.0; PetscReal y_0 = 0.0; PetscReal z_0 = 0.0;
	PetscReal r = 0.5; 
	
	if ((x-x_0)*(x-x_0) + (y-y_0)*(y-y_0) + (z-z_0)*(z-z_0) <= r*r ) {
		
		V0.comp[0] = 1000.0;
		V0.comp[1] = 0.0;
		V0.comp[2] = 0.0; 
        V0.comp[3] = 0.0;
		V0.comp[4] = 1.0;
		V0.comp[5] = 0.99; 
	}
    */

/*
    PetscReal R0 = 1.0;
    PetscReal x0 = 0.0; const PetscReal y0 = 0.0; const PetscReal z0 = 0.0; 
    PetscReal R = PetscSqrtReal((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0)); 

    // Inside the bubble 
    PetscReal rhoi   = 1.0e-2;
    PetscReal pi     = 1.0;
    PetscReal alphai = 0.0;

    // Outside the bubble
    PetscReal rhoo   = 1.0;
    PetscReal po     = 2.0;
    PetscReal alphao = 1.0;

    PetscReal h = 0.05;
    PetscReal smear = 2.0; 

    V0.comp[0] = 0.5*((rhoi+rhoo) + (rhoo-rhoi)*PetscTanhReal((R-R0)/(smear*h)));
    V0.comp[1] = 0.0; 
    V0.comp[2] = 0.0; 
    V0.comp[3] = 0.0; 
    V0.comp[4] = 0.5*((pi+po) + (po-pi)*PetscTanhReal((R-R0)/(smear*h)));
    V0.comp[5] = 0.5*((alphai+alphao) + (alphao-alphai)*PetscTanhReal((R-R0)/(smear*h)));     
*/
    
    //----------------------------------------------------------------------- 

    PDEPrim2Cons(&V0, &Q0);

    return Q0; 
}

/* Test Cases 

*/
