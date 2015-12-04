from model import Model

import numpy as np

class IzhikevichModel(Model):
    def __init__(self, parameters=None):
        Model.__init__(self, parameters)

        self.a = 0.02
        self.b = 0.2
        self.c = -65
        self.d = 8

        t_end = 100
        self.dt = 0.25

        self.v0 = -70

        self.t = np.linspace(0, t_end, t_end/self.dt)

    def I(self, t):
        if 10 <= t:
            return 10
        else:
            return 0

    def f(self, u_in, t):
        v, u = u_in

        dvdt = 0.04*v**2 + 5*v + 140 - u + self.I(t)
        dudt = self.a*(self.b*v - u)

        return np.array([dvdt, dudt])

    def run(self):

        u = np.zeros((len(self.t), 2))
        u[0, 0] = self.v0
        u[0, 1] = self.b*self.v0

        self.U = [self.v0]*len(self.t)

        dt2 = self.dt/2.0
        for n in xrange(len(self.t) - 1):
            K1 = self.dt*self.f(u[n], self.t[n])
            K2 = self.dt*self.f(u[n] + 0.5*K1, self.t[n] + dt2)
            K3 = self.dt*self.f(u[n] + 0.5*K2, self.t[n] + dt2)
            K4 = self.dt*self.f(u[n] + K3, self.t[n] + self.dt)
            u_new = u[n] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)

            if u_new[0] > 30:
                u_new[0] = self.c
                u_new[1] += self.d
                self.U[n] = 30
            else:
                self.U[n] = u_new[0]

            u[n+1] = u_new
