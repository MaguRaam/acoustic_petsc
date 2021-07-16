/*
 * hype.h
 *      Author: sunder
 */

#ifndef HYPE_H_
#define HYPE_H_

//----------------------------------------------------------------------------
// Commonly used C header files
//----------------------------------------------------------------------------

#include <stdio.h>
#include <string.h>

//----------------------------------------------------------------------------
// Petsc headers files 
//----------------------------------------------------------------------------

#include <petscvec.h>
#include <petscmath.h>
#include <petscdm.h>
#include <petscdmplex.h>
#include <petscts.h>
#include <petscdraw.h>
#include <petsctime.h>
#include <petscdmda.h>

//----------------------------------------------------------------------------
// Various constants used throught the code 
//----------------------------------------------------------------------------

#define nVar 2       /* Number of components in the PDE system */ 
#define DIM 2        /* Dimensions of the problem */
#define nDOF 10      /* Number of degrees of freedom polynomial expansion in a cell */

// Physical Constants 
static const PetscReal wave_speed    = 1.0;      /* Wave speed */



static const PetscReal prs_floor     = 1.0e-12; /* Pressure floor value */
static const PetscReal rho_floor     = 1.0e-14; /* Density floor value */
static const PetscReal small_num     = 1.0e-12; /* Effective small number in the code */
static const PetscInt  s_width       = 5;       /* Width of the stencil */ 

// Mid-point Rule  (One-point gauss quadrature)

static const PetscInt N_gp1 = 1; 
static const PetscReal x_gp1[] = {0.0}; 
static const PetscReal w_gp1[] = {1.0}; 

// Two-point quadrature  

static const PetscInt N_gp2 = 2; 
static const PetscReal x_gp2[] = {-0.28867513459481287, 0.28867513459481287}; 
static const PetscReal w_gp2[] = { 0.50000000000000000, 0.50000000000000000}; 

// Three-point quadrature

static const PetscInt N_gp3 = 3; 
static const PetscReal x_gp3[] = {0.00000000000000000, -0.3872983346207417, 0.3872983346207417}; 
static const PetscReal w_gp3[] = {0.44444444444444444,  0.2777777777777778, 0.2777777777777778}; 

// Four-point quadrature

static const PetscInt N_gp4 = 4; 
static const PetscReal x_gp4[] = {0.1699905217924282, -0.1699905217924282, 0.4305681557970263, -0.4305681557970263}; 
static const PetscReal w_gp4[] = {0.3260725774312730,  0.3260725774312730, 0.1739274225687270,  0.1739274225687270};

// Five-point quadrature

static const PetscInt N_gp5 = 5; 
static const PetscReal x_gp5[] = { 0.0000000000000000, -0.2692346550528416,  0.2692346550528416, -0.4530899229693320,  0.4530899229693320}; 
static const PetscReal w_gp5[] = {0.28444444444444444, 0.23931433524968326, 0.23931433524968326, 0.11846344252809456, 0.11846344252809456};

// 3 - point quadrature points in [0,1] for path integrals  

static const PetscInt N_gps = 3;
static const PetscReal s_gp[] = {0.1127016653792583, 0.5, 0.8872983346207417};
static const PetscReal w_gp[] = {0.2777777777777778, 0.4444444444444444, 0.2777777777777778};

// 2D - two point quadrature points in [-0.5,0.5]x[-0.5,0.5]

static const PetscInt N_gp2d = 4;
static const PetscReal x_gp2d[] = {-0.28867513459481287, -0.28867513459481287,  0.28867513459481287, 0.28867513459481287};
static const PetscReal y_gp2d[] = {-0.28867513459481287,  0.28867513459481287, -0.28867513459481287, 0.28867513459481287};
static const PetscReal w_gp2d[] = {0.25,  0.25, 0.25, 0.25};



// Some rational numbers frequently used throught the code

static const PetscReal r1_6  = 1./6.; 
static const PetscReal r13_3 = 13./3.; 
static const PetscReal r7_6  = 7./6.;  
static const PetscReal r11_120 = 11./120.;
static const PetscReal r1_12 = 1./12.;  
static const PetscReal r41_60 = 41./60.; 

//----------------------------------------------------------------------------
// Structure representing a multi-component field vector 
//----------------------------------------------------------------------------

typedef struct {
    PetscReal comp[nVar];
} Field;


//----------------------------------------------------------------------------
// Various types of boundary conditions 
//----------------------------------------------------------------------------

enum bndry_type{inflow, periodic, adiabatic_wall, transmissive};

//----------------------------------------------------------------------------
// Multidimensional array structures (upto 7 dimensions)
//---------------------------------------------------------------------------- 

// 1D

typedef struct {
  PetscInt size;
  PetscInt nelem; 
  PetscReal * data;
} array1d;

array1d* allocate1d(PetscInt);
array1d* copy_array1d(array1d*);
PetscInt free1d(array1d*);
void set_element_1d(array1d*, PetscInt, PetscReal);
PetscReal get_element_1d(array1d*, PetscInt);
void min_max_1d(array1d*, PetscReal*, PetscReal*);

// 2D

typedef struct {
  PetscInt size1; // rows
  PetscInt size2; // cols
  PetscInt nelem; 
  PetscReal * data;
} array2d;

array2d* allocate2d(PetscInt, PetscInt);    
array2d * copy_array2d(array2d*);
PetscInt free2d(array2d*);
void set_element_2d(array2d*, PetscInt, PetscInt, PetscReal);
PetscReal get_element_2d(array2d*, PetscInt, PetscInt);
void min_max_2d(array2d*, PetscReal*, PetscReal*);

// 3D

typedef struct {
  PetscInt size1, size2, size3; 
  PetscInt c1, c2, c3;  
  PetscInt nelem; 
  PetscReal * data;
} array3d;

array3d* allocate3d(PetscInt, PetscInt, PetscInt);    
array3d * copy_array3d(array3d*);
PetscInt free3d(array3d*);
void set_element_3d(array3d*, PetscInt, PetscInt, PetscInt, PetscReal);
PetscReal get_element_3d(array3d*, PetscInt, PetscInt, PetscInt);
void min_max_3d(array3d*, PetscReal*, PetscReal*);

// 4D

typedef struct {
  PetscInt size1, size2, size3, size4; 
  PetscInt c1, c2, c3, c4;  
  PetscInt nelem; 
  PetscReal * data;
} array4d;

array4d* allocate4d(PetscInt, PetscInt, PetscInt, PetscInt);
array4d * copy_array4d(array4d*);
PetscInt free4d(array4d*);
void set_element_4d(array4d*, PetscInt, PetscInt, PetscInt, PetscInt, PetscReal);
PetscReal get_element_4d(array4d*, PetscInt, PetscInt, PetscInt, PetscInt);
void min_max_4d(array4d*, PetscReal*, PetscReal*);

// 5D

typedef struct {
  PetscInt size1, size2, size3, size4, size5; 
  PetscInt c1, c2, c3, c4, c5;  
  PetscInt nelem; 
  PetscReal * data;
} array5d;

array5d* allocate5d(PetscInt, PetscInt, PetscInt, PetscInt, PetscInt);
array5d * copy_array5d(array5d*);
PetscInt free5d(array5d*);
void set_element_5d(array5d*, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscReal);
PetscReal get_element_5d(array5d*, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt);
void min_max_5d(array5d*, PetscReal*, PetscReal*);

// 6D

typedef struct {
  PetscInt size1, size2, size3, size4, size5, size6; 
  PetscInt c1, c2, c3, c4, c5, c6;  
  PetscInt nelem; 
  PetscReal * data;
} array6d;

array6d* allocate6d(PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt);
array6d * copy_array6d(array6d*);
PetscInt free6d(array6d*);
void set_element_6d(array6d*, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscReal);
PetscReal get_element_6d(array6d*, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt);
void min_max_6d(array6d*, PetscReal*, PetscReal*);

// 7D

typedef struct {
  PetscInt size1, size2, size3, size4, size5, size6, size7; 
  PetscInt c1, c2, c3, c4, c5, c6, c7;  
  PetscInt nelem; 
  PetscReal * data;
} array7d;

array7d* allocate7d(PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt);
array7d * copy_array7d(array7d*);
PetscInt free7d(array7d*);
void set_element_7d(array7d*, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscReal);
PetscReal get_element_7d(array7d*, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt);
void min_max_7d(array7d*, PetscReal*, PetscReal*);


//----------------------------------------------------------------------------
// Structure defining various critical parameters controlling the simulation 
//----------------------------------------------------------------------------

typedef struct {
    PetscReal x_min;                  /* x-coordinate of the domain begining */  
    PetscReal y_min;                  /* y-coordinate of the domain begining */
    PetscReal x_max;                  /* x-coordinate of the domain ending */
    PetscReal y_max;                  /* y-coordinate of the domain ending */
    PetscInt N_x;                     /* No. of cells in the x-direction */
    PetscInt N_y;                     /* No. of cells in the y-direction */
    PetscReal CFL;                    /* CFL condition, should be less than 0.5 */
    PetscReal dt;                     /* Time step size */
    PetscReal h;                      /* Grid size */
    PetscBool Restart;                /* Whether to start from restart file */
    PetscInt InitialStep;             /* Initial time step */
    PetscReal InitialTime;            /* Initial time of the simulation */
    PetscReal FinalTime;              /* Final time of the simulation */
    PetscInt WriteInterval;           /* No. of time steps after which data should be written */
    PetscInt RestartInterval;         /* No. of time steps after which restart file should be written */
    enum bndry_type left_boundary;    /* Boundary condition on the left face */
    enum bndry_type right_boundary;   /* Boundary condition on the right face */
    enum bndry_type top_boundary;     /* Boundary condition on the top face */
    enum bndry_type bottom_boundary;  /* Boundary condition on the bottom face */
    Vec localU;                       /* Local solution vector */
    array5d* u_bnd_grad;              /* Boundary extrapolated gradients of conservative variables */
    array2d* F;                       /* Upwind flux in x-direction */
    array2d* G;                       /* Upwind flux in y-direction */
    array3d* gradphiFace_x;           /* Gradient in x-direction on volume quadrature points */
    array3d* gradphiFace_y;           /* Gradient in y-direction on volume quadrature points */
    
} AppCtx;

//----------------------------------------------------------------------------
// Functions related to PDE
//----------------------------------------------------------------------------
	
void PDEFlux(const PetscReal grad_Q[nVar][DIM], PetscReal, PetscReal, PetscReal*);
void LLFRiemannSolver(PetscReal grad_QL[nVar][DIM], PetscReal grad_QR[nVar][DIM], const PetscReal, const PetscReal, PetscReal*);

//----------------------------------------------------------------------------
// WENO reconstruction 
//----------------------------------------------------------------------------

PetscReal basis(PetscReal, PetscReal, PetscInt);
void basis_grad(PetscReal, PetscReal, PetscInt, PetscReal*, PetscReal*); 
void weno(const PetscReal U_x[], const PetscReal U_y[], const PetscReal U_xy[], PetscReal u_coeffs[]);

//----------------------------------------------------------------------------
// Main functions related to the solver 
//----------------------------------------------------------------------------

void InitialCondition(PetscReal, PetscReal, PetscReal*);
void ExactSolution(PetscReal, PetscReal, PetscReal, PetscReal*);
PetscErrorCode InitializeSolution(Vec, DM, AppCtx);
PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void*);
PetscErrorCode MonitorFunction (TS, PetscInt, PetscReal, Vec, void*);
PetscErrorCode ErrorNorms(Vec, DM, AppCtx, PetscReal*, PetscReal*, PetscReal);
PetscErrorCode ComputePrimitiveVariables(Vec, Vec, DM);


#endif /* HYPE_H_ */ 
