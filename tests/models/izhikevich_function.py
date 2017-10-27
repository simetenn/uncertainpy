import numpy as np


def I(t):
    if 10 <= t:
        return 10
    else:
        return 0


def izhikevich(a=0.02, b=0.2, c=-50, d=2):
    t_end = 100
    dt = 0.25

    v0 = -70

    t = np.linspace(0, t_end, t_end/dt)


    def f(u_in, t):
        v, u = u_in

        dvdt = 0.04*v**2 + 5*v + 140 - u + I(t)
        dudt = a*(b*v - u)

        return np.array([dvdt, dudt])

    u = np.zeros((len(t), 2))
    u[0, 0] = v0
    u[0, 1] = b*v0


    U = [v0]*len(t)


    # Runge Kutta 4
    dt2 = dt/2.0
    for n in xrange(len(t) - 1):
        K1 = dt*f(u[n], t[n])
        K2 = dt*f(u[n] + 0.5*K1, t[n] + dt2)
        K3 = dt*f(u[n] + 0.5*K2, t[n] + dt2)
        K4 = dt*f(u[n] + K3, t[n] + dt)
        u_new = u[n] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)

        if u_new[0] > 30:
            u_new[0] = c
            u_new[1] += d
            U[n] = 30
        else:
            U[n] = u_new[0]

        u[n+1] = u_new

    return t, np.array(U)
