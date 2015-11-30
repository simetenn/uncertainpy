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
    def __init__(self):
        """
        Init must be able to be called with 0 arguments
        """
        Model.__init__(self)

        ## Functions
        # K channel
        self.alpha_n = vectorize(lambda v: 0.01*(-v + 10)/(exp((-v + 10)/10) - 1) if v != 10 else 0.1)
        self.beta_n = lambda v: 0.125*exp(-v/80)
        self.n_inf = lambda v: self.alpha_n(v)/(self.alpha_n(v) + self.beta_n(v))

        # Na channel (activating)
        self.alpha_m = vectorize(lambda v: 0.1*(-v + 25)/(exp((-v + 25)/10) - 1) if v != 25 else 1)
        self.beta_m = lambda v: 4*exp(-v/18)
        self.m_inf = lambda v: self.alpha_m(v)/(self.alpha_m(v) + self.beta_m(v))

        # Na channel (inactivating)
        self.alpha_h = lambda v: 0.07*exp(-v/20)
        self.beta_h = lambda v: 1/(exp((-v + 30)/10) + 1)
        self.h_inf = lambda v: self.alpha_h(v)/(self.alpha_h(v) + self.beta_h(v))

        ### channel activity ###
        self.v = arange(-50, 151)  # mV


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



    def run(self):
        Vm = zeros(len(self.time))  # mV
        Vm[0] = self.V_rest
        m = self.m_inf(V_rest)
        h = self.h_inf(V_rest)
        n = self.n_inf(V_rest)

        for i in range(1, len(self.time)):
            g_Na = self.gbar_Na*(m**3)*h
            g_K = self.gbar_K*(n**4)
            g_l = self.gbar_l

            m += dt*(self.alpha_m(Vm[i-1])*(1 - m) - self.beta_m(Vm[i-1])*m)
            h += dt*(self.alpha_h(Vm[i-1])*(1 - h) - self.beta_h(Vm[i-1])*h)
            n += dt*(self.alpha_n(Vm[i-1])*(1 - n) - self.beta_n(Vm[i-1])*n)

            Vm[i] = Vm[i-1] + (I[i-1] - g_Na*(Vm[i-1] - E_Na) - g_K*(Vm[i-1] - E_K) - g_l*(Vm[i-1] - E_l)) / Cm * dt

        self.U = Vm


    def setParameters(self, parameters):
        """
        Parameters: dictionary with all parameters
        """
        # How the parameters are set
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter]))
