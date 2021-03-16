import numpy
from matplotlib import pyplot, cm
from math import log

def convergence_plot(Ngpts,l2,linfty,filename):
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



N  = numpy.array([16,32,64,128,256])
linfty = numpy.array([1.4998953e-03, 9.9664700e-05,6.3248623e-06,3.9681436e-07,2.4824543e-08])  
l2     = numpy.array([7.7962017e-04,5.0315752e-05,3.1700635e-06,1.9852674e-07,1.2414137e-08])
convergence_plot(N,l2,linfty,"convergence")

#L2 rate of convergence
print("L2 Rate of convergence:")
convergence_rate(l2)

#linfty rate of convergence:
print("Linfty Rate of convergence")
convergence_rate(linfty)