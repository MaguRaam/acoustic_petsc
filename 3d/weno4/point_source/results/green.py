#Numerical solution vs anlytical solution obtained by convolving Green's function with the time dependent source function

import numpy as np
import matplotlib.pyplot as plt

#Set simulation parameter same as your acoustic petsc code:

#grid spacing
dx             =   0.5
dy             =   0.5
dz             =   0.5
c0             =   250.0
dt             =   0.0005
nt             =   251

#source and reciever location
isx            =   50
isy            =   50
isz            =   50
ts             =   0.0

irx            =   57
iry            =   57
irz            =   57

#source frequency
f0 = 100.0
t0 = 4.0/ f0;


#distance between source and reciever
r = np.sqrt((isx*dx - irx*dx) ** 2 + (isy*dy - iry*dy) ** 2 + (isz*dz - irz*dz) ** 2)

#time axis
time = np.linspace(0 * dt, nt * dt, nt)

#source function
src  = -2. * (time - t0) * (f0 ** 2) * (np.exp(-1.0 * (f0 ** 2) * (time - t0) ** 2))

#compute Greens function for 3d acoustic wave equation
G    = time * 0.
t_ar = time[0] + r / c0
i_ar = int(t_ar / dt)
G[i_ar] = 1.0 / (4 * np.pi * c0 ** 2 * r) # Calculate Green's function


#compute analytical solution by convolving Green's function with the source function
Analytical = np.convolve(G, src)
Analytical = Analytical[0:nt]


#plot Green's function
plt.figure(figsize=(9,9))
plt.plot(time, G)
plt.title("Green's function in 3D")
plt.xlabel('Time, s')
plt.ylabel('Amplitude')
plt.grid()
plt.savefig("Green")


#plot source time function
plt.figure(figsize=(9,9))
plt.plot(time, src)
plt.title('Source time function')
plt.xlabel('Time, s')
plt.ylabel('Amplitude')
plt.grid()
plt.savefig("Source")



#plot Numerical and Analytical Pressure as a function of time at reciever (irx, iry, irz)
p_t = np.loadtxt("p.dat")

fig,ax = plt.subplots()
plt.figure(figsize=(9,9))
plt.title('Pressure as a function of time at grid point (irx, iry, irz)')
plt.xlabel('Time, s')
plt.ylabel('Pressure')
leg1,= plt.plot(p_t[:,0], p_t[:,1],'b-',markersize=1)
leg2,= plt.plot(time, Analytical,'r--',markersize=1)
plt.legend((leg1, leg2), ('Numerical', 'Analytical'), loc='upper right', fontsize=10, numpoints=1)
plt.savefig("Pressure")
