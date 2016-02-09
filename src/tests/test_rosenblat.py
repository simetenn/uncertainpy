import chaospy as cp
import numpy as np
import pylab as plt

def u(x, a, I):
    return I*np.exp(-a*x)


x = np.linspace(0, 10, 10)
dt = 10/1000.

dist_R = cp.J(cp.Uniform(-1, 1), cp.Uniform(-1, 1))
dist_Q = cp.J(cp.Uniform(0.1), cp.Uniform(8, 10))

M = 4

P = cp.orth_ttr(M, dist_R)
s_R = dist_R.sample(2*len(P), "M")
s_Q = dist_Q.inv(dist_R.fwd(s_R))
print s_R
print s_Q



solves = [u(x, s[0], s[1]) for s in s_Q.T]
U_hat = cp.fit_regression(P, s_R, solves, rule="LS")



E = cp.E(U_hat, dist_R)

plt.plot(x, E)
plt.show()
