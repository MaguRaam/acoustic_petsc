#Numerical solution vs anlytical solution obtained by convolving Green's function with the time dependent source function

import numpy as np
import matplotlib.pyplot as plt

#Set simulation parameter same as your acoustic petsc code:

#grid spacing
dx             =   1.0 
dy             =   1.0
c0             =   580.0
dt             =   0.0010
nt             =   502

#source and reciever location
isx            =   250
isy            =   250
ts             =   0.0

irx            =   330
iry            =   330

#source frequency
f0 = 65.0
t0 = 4.0/ f0;


#distance between source and reciever
r = np.sqrt((isx*dx - irx*dx) ** 2 + (isy*dy - iry*dy) ** 2)

#time axis
time = np.linspace(0 * dt, nt * dt, nt)

#source function
src  = -2. * (time - t0) * (f0 ** 2) * (np.exp(-1.0 * (f0 ** 2) * (time - t0) ** 2))

#compute Greens function for 2d acoustic wave equation
G    = time * 0.
for it in range(nt):
    if (((time[it]-ts) - np.abs(r) / c0) >= 0):
        G[it] = (1. / (2 * np.pi * c0 ** 2)) * (1. / np.sqrt(((time[it]-ts) ** 2) - (r ** 2 / (c0 ** 2))))
    else:
        G[it] = 0.0

#compute analytical solution by convolving Green's function with the source function
Analytical = np.convolve(G, src * dt)



#plot Green's function
plt.figure(figsize=(9,9))
plt.plot(time, G)
plt.title("Green's function in 2D")
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



#plot Numerical and Analytical Pressure as a function of time at reciever (irx, iry)
p_t = np.loadtxt("p.dat")

fig,ax = plt.subplots()
plt.figure(figsize=(9,9))
plt.title('Pressure as a function of time at grid point (irx, iry)')
plt.xlabel('Time, s')
plt.ylabel('Pressure')
leg1,= plt.plot(p_t[:,0], p_t[:,1],'b-',markersize=1)
leg2,= plt.plot(time, Analytical[0:nt],'r--',markersize=1)
plt.legend((leg1, leg2), ('Numerical', 'Analytical'), loc='upper right', fontsize=10, numpoints=1)
plt.savefig("Pressure")
