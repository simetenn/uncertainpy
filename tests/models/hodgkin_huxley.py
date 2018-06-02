from uncertainpy import Model

import numpy as np
from scipy.integrate import odeint



class HodgkinHuxley(Model):
    def __init__(self):
        Model.__init__(self,
                       interpolate=False,
                       labels=["Time (ms)", "Voltage (mV)"],
                       logger_level=None)

        ## HH Parameters
        self.V_rest = -65   # mV
        self.Cm = 1         # uF/cm**2
        self.gbar_Na = 120  # mS/cm**2
        self.gbar_K = 36    # mS/cm**2
        self.gbar_L = 0.3   # mS/cm**2
        self.E_Na = 50      # mV
        self.E_K = -77      # mV
        self.E_l = -54.4    # mV


        ## setup parameters and state variables
        self.I_value = 150
        T = 15    # ms
        dt = 0.025  # ms
        self.time = np.arange(0, T + dt, dt)



    def I(self, t):
        return self.I_value

    # K channel
    def alpha_n(self, V):
        return 0.01*(V + 55)/(1 - np.exp(-(V + 55)/10.))

    def beta_n(self, V):
        return 0.125*np.exp(-(V + 65)/80.)

    def n_f(self, n, V):
        return self.alpha_n(V)*(1 - n) - self.beta_n(V)*n

    def n_inf(self, V):
        return self.alpha_n(self.V_rest)/(self.alpha_n(self.V_rest) + self.beta_n(self.V_rest))


    # Na channel (activating)
    def alpha_m(self, V):
        return 0.1*(V + 40)/(1 - np.exp(-(V + 40)/10.))

    def beta_m(self, V):
        return 4*np.exp(-(V + 65)/18.)

    def m_f(self, m, V):
        return self.alpha_m(V)*(1 - m) - self.beta_m(V)*m

    def m_inf(self, V):
        return self.alpha_m(self.V_rest)/(self.alpha_m(self.V_rest) + self.beta_m(self.V_rest))


    # Na channel (inactivating)
    def alpha_h(self, V):
        return 0.07*np.exp(-(V + 65)/20.)

    def beta_h(self, V):
        return 1/(np.exp(-(V + 35)/10.) + 1)

    def h_f(self, h, V):
        return self.alpha_h(V)*(1 - h) - self.beta_h(V)*h

    def h_inf(self, V):
        return self.alpha_h(self.V_rest)/(self.alpha_h(self.V_rest) + self.beta_h(self.V_rest))


    def dXdt(self, X, t):
        V, h, m, n = X

        g_Na = self.gbar_Na*(m**3)*h
        g_K = self.gbar_K*(n**4)
        g_l = self.gbar_L

        dmdt = self.m_f(m, V)
        dhdt = self.h_f(h, V)
        dndt = self.n_f(n, V)

        dVdt = (self.I(t) - g_Na*(V - self.E_Na) - g_K*(V - self.E_K) - g_l*(V - self.E_l))/self.Cm

        return [dVdt, dhdt, dmdt, dndt]


    def run(self, **parameters):
        self.set_parameters(**parameters)

        self.h0 = self.h_inf(self.V_rest)
        self.m0 = self.m_inf(self.V_rest)
        self.n0 = self.n_inf(self.V_rest)

        initial_conditions = [self.V_rest, self.h0, self.m0, self.n0]

        X = odeint(self.dXdt, initial_conditions, self.time)
        values = X[:, 0]

         # Add info needed by certain spiking features and efel features
        info = {"stimulus_start": self.time[0], "stimulus_end": self.time[-1]}

        return self.time, values, info
