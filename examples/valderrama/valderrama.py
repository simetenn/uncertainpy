from uncertainpy import Model

import numpy as np
from scipy.integrate import odeint

class Valderrama(Model):
    """

    """
    def __init__(self):
        Model.__init__(self,
                       labels=["Time (ms)", "Membrane potential (mV)"])


        ## HH Parameters
        self.V_0 = -10      # mV
        self.C_m = 1        # uF/cm**2
        self.gbar_Na = 120  # mS/cm**2
        self.gbar_K = 36    # mS/cm**2
        self.gbar_l = 0.3   # mS/cm**2
        self.E_Na = 115     # mV
        self.E_K = -12      # mV
        self.E_l = 10.6     # mV

        self.m_0 = 0.0011    # unitless
        self.n_0 = 0.0003    # unitless
        self.h_0 = 0.9998    # unitless

        ## setup parameters and state variables
        self.I_value = 150     # mA
        T = 15                 # ms
        dt = 0.025             # ms
        self.time = np.arange(0, T + dt, dt)


    def I(self, t):
        return self.I_value

    # K channel
    def alpha_n(self, V):
        return 0.01*(10 - V)/(np.exp((10 - V)/10.) - 1)

    def beta_n(self, V):
        return 0.125*np.exp(-V/80.)

    def n_f(self, n, V):
        return self.alpha_n(V)*(1 - n) - self.beta_n(V)*n

    def n_inf(self, V):
        return self.alpha_n(V)/(self.alpha_n(V) + self.beta_n(V))


    # Na channel (activating)
    def alpha_m(self, V):
        return 0.1*(25 - V)/(np.exp((25 - V)/10.) - 1)

    def beta_m(self, V):
        return 4*np.exp(-V/18.)

    def m_f(self, m, V):
        return self.alpha_m(V)*(1 - m) - self.beta_m(V)*m

    def m_inf(self, V):
        return self.alpha_m(V)/(self.alpha_m(V) + self.beta_m(V))


    # Na channel (inactivating)
    def alpha_h(self, V):
        return 0.07*np.exp(-V/20.)

    def beta_h(self, V):
        return 1/(np.exp((30 - V)/10.) + 1)

    def h_f(self, h, V):
        return self.alpha_h(V)*(1 - h) - self.beta_h(V)*h

    def h_inf(self, V):
        return self.alpha_h(V)/(self.alpha_h(V) + self.beta_h(V))


    def dXdt(self, X, t):
        V, h, m, n = X

        g_Na = self.gbar_Na*(m**3)*h
        g_K = self.gbar_K*(n**4)
        g_l = self.gbar_l

        dmdt = self.m_f(m, V)
        dhdt = self.h_f(h, V)
        dndt = self.n_f(n, V)

        dVdt = (self.I(t) - g_Na*(V - self.E_Na) - g_K*(V - self.E_K) - g_l*(V - self.E_l))/self.C_m

        return [dVdt, dhdt, dmdt, dndt]


    def run(self, **parameters):
        self.set_parameters(**parameters)

        initial_conditions = [self.V_0, self.h_0, self.m_0, self.n_0]

        X = odeint(self.dXdt, initial_conditions, self.time)
        values = X[:, 0]

        time = self.time
        values = X[:, 0]

        # Only return from 5 seconds onwards, as in valderamma
        values = values[time > 5]
        time = time[time > 5]

        # Add info needed by certain spiking features and efel features
        info = {"stimulus_start": time[0], "stimulus_end": time[-1]}

        return time, values, info

if __name__ == "__main__":
    model = Valderrama()
    model.run()
