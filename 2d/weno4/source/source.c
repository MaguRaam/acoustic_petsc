/*
 * initial_condition.c
 *      Author: sunder
 */ 
#include "hype.h" 

//----------------------------------------------------------------------------
// Source as a function of time
//----------------------------------------------------------------------------


PetscReal Source(PetscReal  t){

    PetscReal   f0 = 65.0;      /*dominant frequency is 40 Hz*/
    PetscReal   t0 = 4.0/ f0;   /*source time shift*/
    
    return -2.0*(t - t0)*f0*f0*PetscExpReal( -1.0*f0*f0*(t - t0)*(t - t0));
}