from model import Model

import numpy as np

# The class name and file name must be the same
class HodkinHuxleyModel(Model):
    """
    The model must be able to handle these calls

    simulation = model()
    simulation.load()
    simulation.setParameters(parameters -> dictionary)
    simulation.run()
    simulation.save(current_process -> int)

    simulation.cmd()
    """
    def __init__(self, parameters=None):
        """
        Init must be able to be called with 0 arguments
        """
        Model.__init__(self, parameters=parameters)

        ## Functions
        # K channel
        self.alpha_n = np.vectorize(lambda v: 0.01*(-v + 10)/(np.exp((-v + 10)/10) - 1) if v != 10 else 0.1)
        self.beta_n = lambda v: 0.125*np.exp(-v/80)
        self.n_inf = lambda v: self.alpha_n(v)/(self.alpha_n(v) + self.beta_n(v))

        # Na channel (activating)
        self.alpha_m = np.vectorize(lambda v: 0.1*(-v + 25)/(np.exp((-v + 25)/10) - 1) if v != 25 else 1)
        self.beta_m = lambda v: 4*np.exp(-v/18)
        self.m_inf = lambda v: self.alpha_m(v)/(self.alpha_m(v) + self.beta_m(v))

        # Na channel (inactivating)
        self.alpha_h = lambda v: 0.07*np.exp(-v/20)
        self.beta_h = lambda v: 1/(np.exp((-v + 30)/10) + 1)
        self.h_inf = lambda v: self.alpha_h(v)/(self.alpha_h(v) + self.beta_h(v))

        ## setup parameters and state variables
        self.T = 45    # ms
        self.dt = 0.025  # ms
        self.t = np.arange(0, self.T + self.dt, self.dt)

        ## HH Parameters
        self.V_rest = 0     # mV
        self.Cm = 1         # uF/cm2
        self.gbar_Na = 120  # mS/cm2
        self.gbar_K = 36    # mS/cm2
        self.gbar_l = 0.3   # mS/cm2
        self.E_Na = 115     # mV
        self.E_K = -12      # mV
        self.E_l = 10.613   # mV


        self.I = np.zeros(len(self.t))
        for i, t in enumerate(self.t):
            if 5 <= t <= 30:
                self.I[i] = 10  # uA/cm2


        self.xlabel = "time [ms]"
        self.ylabel = "voltage [mv]"

    def run(self):

        Vm = np.zeros(len(self.t))  # mV
        Vm[0] = self.V_rest
        m = self.m_inf(self.V_rest)
        h = self.h_inf(self.V_rest)
        n = self.n_inf(self.V_rest)

        for i in range(1, len(self.t)):
            g_Na = self.gbar_Na*(m**3)*h
            g_K = self.gbar_K*(n**4)
            g_l = self.gbar_l

            m += self.dt*(self.alpha_m(Vm[i-1])*(1 - m) - self.beta_m(Vm[i-1])*m)
            h += self.dt*(self.alpha_h(Vm[i-1])*(1 - h) - self.beta_h(Vm[i-1])*h)
            n += self.dt*(self.alpha_n(Vm[i-1])*(1 - n) - self.beta_n(Vm[i-1])*n)

            Vm[i] = Vm[i-1] + (self.I[i-1] - g_Na*(Vm[i-1] - self.E_Na) - g_K*(Vm[i-1] - self.E_K) - g_l*(Vm[i-1] - self.E_l))/self.Cm*self.dt

        self.U = Vm - 65
