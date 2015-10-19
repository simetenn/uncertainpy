import chaospy as cp
import numpy as np
import pylab as plt

def u(x, a, I):
    return I*np.exp(-a*x)

def E_analytical(x):
    return 90*(1-np.exp(-0.1*x))/(x)

a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a, I)

x = np.linspace(0, 10, 100)
M = 4

P = cp.orth_ttr(M, dist)
nodes, weights = cp.generate_quadrature(M + 1, dist, rule="P", sparse=True)
#nodes = dist.sample(2*len(P), "G")
print nodes.shape
solves = [u(x, *s) for s in nodes.T]
U_hat = cp.fit_quadrature(P, nodes, weights, solves)
#U_hat = cp.fit_regression(P, nodes, solves, rule="T")


plt.plot(x, cp.E(U_hat, dist), linewidth=2)
plt.plot(x, E_analytical(x), linewidth=2)
plt.legend(["PC", "Analytical"])
plt.show()
