from uncertainpy import Model

import numpy as np

class Izhikevich(Model):
    def __init__(self):
        Model.__init__(self,
                       labels=["Time [ms]", "Voltage [mV]"])

        time_end = 100
        self.dt = 0.25
        self.v0 = -70
        self.time = np.linspace(0, time_end, time_end/self.dt)

        self.info = {"stimulus_start": 10,
                     "stimulus_end": time_end}

    def I(self, time):
        if time >= 10:
            return 10
        else:
            return 0


    def run(self, a=0.02, b=0.2, c=-50, d=2):

        def f(u_in, time):
            v, u = u_in

            dvdt = 0.04*v**2 + 5*v + 140 - u + self.I(time)
            dudt = a*(b*v - u)

            return np.array([dvdt, dudt])

        u = np.zeros((len(self.time), 2))
        u[0, 0] = self.v0
        u[0, 1] = b*self.v0


        values = [self.v0]*len(self.time)

        # Runge Kutta 4
        dt2 = self.dt/2.0
        for n in xrange(len(self.time) - 1):
            K1 = self.dt*f(u[n], self.time[n])
            K2 = self.dt*f(u[n] + 0.5*K1, self.time[n] + dt2)
            K3 = self.dt*f(u[n] + 0.5*K2, self.time[n] + dt2)
            K4 = self.dt*f(u[n] + K3, self.time[n] + self.dt)
            u_new = u[n] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)

            if u_new[0] > 30:
                u_new[0] = c
                u_new[1] += d
                values[n] = 30
            else:
                values[n] = u_new[0]

            u[n+1] = u_new

        return self.time, np.array(values), self.info
