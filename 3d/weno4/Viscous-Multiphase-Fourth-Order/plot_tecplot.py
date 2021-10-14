import numpy as np 
import os
from datetime import datetime

g1 = 1.4
g2 = 1.6
p1 = 0.0
p2 = 0.0 
dim = 3
n_comp = 6

def conserved_to_primitive(Q):
    V = np.zeros(n_comp)
        
    g = 1.0 + (g1-1.0)*(g2-1.0)/((1.0-Q[5])*(g1 -1.0) + Q[5]*(g2-1.0));
    P_inf = ((g -1.0)/g)*( g1*p1*Q[5]/(g1 - 1.0) + g2*p2*(1.0 - Q[5])/(g2 - 1.0) );
    temp = 1.0/Q[0]; 
    
    V[0] = Q[0];
    V[1] = Q[1]*temp;
    V[2] = Q[2]*temp;
    V[3] = Q[3]*temp; 
    V[4] = (g -1.0)*( Q[4] - 0.5*temp*(Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3]) )  - g*P_inf;
    V[5] = Q[5];
    
    return V; 

def plot_tecplot(in_file, out_file):
    
    infile = open(in_file, "r")
    
    # Read header lines 

    line = infile.readline()
    line = infile.readline()
    line = infile.readline()
    line = infile.readline()
    line = infile.readline()
    split_line = line.split()

    N_x = int(split_line[1])
    N_y = int(split_line[2])
    N_z = int(split_line[3])


    line = infile.readline()

    # Get number of cells in the mesh 

    split_line = line.split()
    n_cells = int(split_line[1])

    U = np.zeros((n_cells,n_comp))
    W = np.zeros((N_z,N_y, N_x, n_comp))

    coords = np.zeros((n_cells,dim))

    # Read the coordinates 

    for i in range(0, n_cells):
        line = infile.readline()
        split_line = line.split()
        coords[i, 0] = np.float64(split_line[0])
        coords[i, 1] = np.float64(split_line[1])
        coords[i, 2] = np.float64(split_line[2])

    # Read some more headers 
        
    line = infile.readline()
    line = infile.readline()
    line = infile.readline()

    for i in range(0, n_cells):
        line = infile.readline()
        split_line = line.split()
        for c in range (0, n_comp):
            U[i,c] = np.float64(split_line[c])
        
    infile.close()

    # Reshape arrays to fit rectangular coordinates 

    coords = coords.reshape((N_z, N_y, N_x, dim))
    U = U.reshape((N_z, N_y, N_x, n_comp))

    # Find the primitive variables

    Q = np.zeros(n_comp)
    V = np.zeros(n_comp)

    for k in range(0, N_z):
        for j in range(0, N_y):
            for i in range (0, N_x):
                for c in range (0, n_comp):
                    Q[c] = U[k,j,i,c]
                V = conserved_to_primitive(Q)
                for c in range (0, n_comp):
                    W[k,j,i,c] = V[c]

    # Start plotting 
            
    outfile = open(out_file, "w+")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
    print("TITLE = \"Effective-Gamma-3D, File created on:", dt_string, "IST\"",  file = outfile)
    outfile.write("\nVARIABLES = \"x\", \"y\", \"z\", \"Density\", \"V_x\", \"V_y\", \"V_z\", \"Pressure\", \"Phi\" ")
    print('\nZone I = ', N_x, "J = ", N_y, "K = ", N_z, file = outfile) 
    
    for k in range(0, N_z):
        for j in range(0, N_y):
            for i in range (0, N_x):
                print(coords[k,j,i,0], coords[k,j,i,1], coords[k,j,i,2], W[k,j,i,0], W[k,j,i,1], W[k,j,i,2], W[k,j,i,3], W[k,j,i,4], W[k,j,i,5], file = outfile)

            
    outfile.close()
###############################################################################################################################

for in_file in os.listdir('.'):
    if in_file.endswith('.vtk'):
        print(in_file)
        out_file = in_file.replace("vtk", "dat")
        plot_tecplot(in_file, out_file)

