import chaospy as cp
import pylab as pl
import numpy as np

Nt = 10**2
N = 10

def E_analytical(x):
    return 90*(1-np.exp(-0.1*x))/(x)


def V_analytical(x):
    return 1220*(1-np.exp(-0.2*x))/(3*x) - (90*(1-np.exp(-0.1*x))/(x))**2


def u(x,a, I):
    return I*np.exp(-a*x)

a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a,I)

T = np.linspace(0, 10, Nt+1)[1:]
dt = 10./Nt


error = []
var = []

K = []
for n in range(2,N):

    P = cp.orth_ttr(n, dist)
    #nodes = dist.sample(2*len(P))
    nodes, weights = cp.generate_quadrature(3, dist, rule="Z", sparse=True)

    K.append(2*len(P))
    #solves = [u(T, s[0], s[1]) for s in nodes.T]
    solves = []
    for s in nodes.T:
        u_ = u(T,*s)
        solves.append(u_)

    #U_hat = cp.fit_regression(P, nodes, solves)
    U_hat = cp.fit_quadrature(P, nodes, weights, solves)

    error.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))

pl.rc("figure", figsize=[6,4])

pl.plot(K,error,"r-",linewidth=2)
pl.plot(K, var,"r--",linewidth=2)
pl.xlabel("Nodes, K")
pl.ylabel("Error")
pl.yscale('log')
pl.xlim([7,110])
pl.title("Error in expectation value and variance ")
pl.legend(["Mean","Variance"])

pl.show()
