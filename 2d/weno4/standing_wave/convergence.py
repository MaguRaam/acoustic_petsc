import numpy
from matplotlib import pyplot, cm
from math import log



def convergence_plot(N,l2,linfty,filename):
  fig,ax = pyplot.subplots()
  pyplot.loglog(N, linfty,'-b',label = 'Linfty error',marker = 'o')
  pyplot.loglog(N, l2,'--r',label = 'L2 error',marker = 'o')
  leg = ax.legend()
  pyplot.title("Convergence in space")
  pyplot.xlabel("N")
  pyplot.ylabel("Error")
  pyplot.savefig(filename)

def convergence_rate(error):
  for i in range(1,numpy.size(error)):
    covergence_rate = (log(error[i-1]) - log(error[i]))/log(2)
    print(covergence_rate)



#load N, l2 and linfty
data = numpy.loadtxt("Error.dat")
convergence_plot(data[:,0],data[:,1],data[:,2],"error")

#L2 rate of convergence
print("L2 Rate of convergence:")
convergence_rate(data[:,1])

#linfty rate of convergence:
print("Linfty Rate of convergence")
convergence_rate(data[:,2])
