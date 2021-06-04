/*
 * initial_condition.c
 *      Author: sunder
 */ 
#include "hype.h" 

//----------------------------------------------------------------------------
// Common Initial Conditions 
//----------------------------------------------------------------------------

void air_helium(PetscReal,PetscReal,PetscReal*);
void air_R22(PetscReal,PetscReal,PetscReal*);
void air_water(PetscReal,PetscReal,PetscReal*);
void water_air(PetscReal,PetscReal,PetscReal*);
void bubble_collapse(PetscReal,PetscReal,PetscReal*);
void richtmyer_meshkov(PetscReal,PetscReal,PetscReal*);

//----------------------------------------------------------------------------
// Initial condition function 
//----------------------------------------------------------------------------

void InitialCondition(PetscReal x, PetscReal y, PetscReal* Q0) {
    
    bubble_collapse(x,y,Q0); 

}

//----------------------------------------------------------------------------
// Air Helium-Bubble Shock Interaction 
// [x,y] \in [0.0,0.356] x [0.0,0.089]
// Final Time: 674.0e-6
// BC: L-T, R-T, B-R, T-R
// g1 = 1.4; g2 = 1.648; p1 = 0.0; g1 = 0.0 
//----------------------------------------------------------------------------

void air_helium(PetscReal x, PetscReal y, PetscReal* Q0) {
    
    PetscReal V0[nVar];

    PetscReal x0 = 0.245, y0 = 0.0455, R = 0.025;
    
    // (Air) Shock 
    
    if (x >= 0.275) {
        V0[0] = 1.92691;
        V0[1] = -114.42;
        V0[2] = 0.0;
        V0[3] = 1.56980e5;
        V0[4] = 1.0;
    }
    
    else {
        V0[0] = 1.4;
        V0[1] = 0.0;
        V0[2] = 0.0;
        V0[3] = 1.0e5;
        V0[4] = 1.0;
    }
    
    // (Helium) Bubble 
    
    if ((x-x0)*(x-x0) + (y-y0)*(y-y0) <= R*R) {
        V0[0] = 0.25463;
        V0[1] = 0.0;
        V0[2] = 0.0;
        V0[3] = 1.0e5;
        V0[4] = 0.0;
    }
    
    PDEPrim2Cons(V0, Q0);
}

//----------------------------------------------------------------------------
// Air R22-Bubble Shock Interaction 
// [x,y] \in [0.0,0.356] x [0.0,0.089]
// Final Time: 417.0e-6
// BC: L-T, R-T, B-R, T-R
// g1 = 1.4; g2 = 1.249; p1 = 0.0; g1 = 0.0 
//----------------------------------------------------------------------------

void air_R22(PetscReal x, PetscReal y, PetscReal* Q0) {
    
    PetscReal V0[nVar];

    PetscReal x0 = 0.245, y0 = 0.0455, R = 0.025;
    
    // (Air) Shock 
    
    if (x >= 0.275) {
        V0[0] = 1.92691;
        V0[1] = -114.42;
        V0[2] = 0.0;
        V0[3] = 1.56980e5;
        V0[4] = 1.0;
    }
    
    else {
        V0[0] = 1.4;
        V0[1] = 0.0;
        V0[2] = 0.0;
        V0[3] = 1.0e5;
        V0[4] = 1.0;
    }
    
    // (R22) Bubble 
    
    if ((x-x0)*(x-x0) + (y-y0)*(y-y0) <= R*R) {
        V0[0] = 4.41540;
        V0[1] = 0.0;
        V0[2] = 0.0;
        V0[3] = 1.0e5;
        V0[4] = 0.0;
    }
    
    PDEPrim2Cons(V0, Q0);
}

//----------------------------------------------------------------------------
// Shock in water hitting air coloumn  
// [x,y] \in [0,10] x [-2.5,2.5]
// Final Time: 0.02
// BC: L-T, R-T, B-T, T-T
// g1 = 4.4; g2 = 1.4; p1 = 6000.0; g1 = 0.0 
//----------------------------------------------------------------------------

void water_air(PetscReal x, PetscReal y, PetscReal* Q0) {
    
    PetscReal V0[nVar]; 
    
    PetscReal x0 = 4.375; PetscReal y0 = 0.0; 
    PetscReal R = 1.0; 
    
    // (Water) Shock 
    
    if (x < 1.0) {
        V0[0] = 1.325;
        V0[1] = 68.52;
        V0[2] = 0.0;
        V0[3] = 1.915e4;
        V0[4] = 1.0;
    }
    
    else {
        V0[0] = 1.0;
        V0[1] = 0.0;
        V0[2] = 0.0;
        V0[3] = 1.0;
        V0[4] = 1.0;
    }
    
    // (Air) Bubble 
    
    if ((x-x0)*(x-x0) + (y-y0)*(y-y0) < R*R) {
        
        V0[0] = 0.001;
        V0[1] = 0.0;
        V0[2] = 0.0;
        V0[3] = 1.0;
        V0[4] = 0.0;
    }
    
    PDEPrim2Cons(V0, Q0);
}

//----------------------------------------------------------------------------
// Bubble Collapse Near Wall 
// [x,y] \in [0,10] x [-2.5,2.5]
// Final Time: 0.02
// BC: L-R, R-T, B-T, T-T
// g1 = 4.4; g2 = 1.4; p1 = 6000.0; g1 = 0.0 
//----------------------------------------------------------------------------

void bubble_collapse(PetscReal x, PetscReal y, PetscReal* Q0) {
    
    PetscReal V0[nVar]; 
    
    PetscReal R0 = 1.0;
    PetscReal x0 = 0.0; const PetscReal y0 = 0.0;
    PetscReal R = PetscSqrtReal((x-x0)*(x-x0) + (y-y0)*(y-y0));

    // Inside the bubble 
    PetscReal rhoi   = 1.0e-2;
    PetscReal pi     = 1.0;
    PetscReal alphai = 0.0;

    // Outside the bubble
    PetscReal rhoo   = 1.0;
    PetscReal po     = 100.0;
    PetscReal alphao = 1.0;

    PetscReal h = 10.0/500.0;
    PetscReal smear = 2.0;

    V0[0] = 0.5*((rhoi+rhoo) + (rhoo-rhoi)*PetscTanhReal((R-R0)/(smear*h)));
    V0[1] = 0.0;
    V0[2] = 0.0;
    V0[3] = 0.5*((pi+po) + (po-pi)*PetscTanhReal((R-R0)/(smear*h)));
    V0[4] = 0.5*((alphai+alphao) + (alphao-alphai)*PetscTanhReal((R-R0)/(smear*h)));

    
    PDEPrim2Cons(V0, Q0);
}

