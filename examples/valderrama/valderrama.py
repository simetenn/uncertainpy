import uncertainpy as un

import numpy as np
from scipy.integrate import odeint


# External stimulus
def I(time):
    return 140 # micro A/cm**2


def valderrama(V_0=-10,
               C_m=1,
               gbar_Na=120,
               gbar_K=36,
               gbar_L=0.3,
               E_Na=112,
               E_K=-12,
               E_l=10.613,
               m_0=0.0011,
               n_0=0.0003,
               h_0=0.9998):

    # Setup time
    end_time = 15          # ms
    dt = 0.025             # ms
    time = np.arange(0, end_time + dt, dt)

    # K channel
    def alpha_n(V):
        return 0.01*(10 - V)/(np.exp((10 - V)/10.) - 1)


    def beta_n(V):
        return 0.125*np.exp(-V/80.)

    def n_f(n, V):
        return alpha_n(V)*(1 - n) - beta_n(V)*n

    def n_inf(V):
        return alpha_n(V)/(alpha_n(V) + beta_n(V))


    # Na channel (activating)
    def alpha_m(V):
        return 0.1*(25 - V)/(np.exp((25 - V)/10.) - 1)

    def beta_m(V):
        return 4*np.exp(-V/18.)

    def m_f(m, V):
        return alpha_m(V)*(1 - m) - beta_m(V)*m

    def m_inf(V):
        return alpha_m(V)/(alpha_m(V) + beta_m(V))


    # Na channel (inactivating)
    def alpha_h(V):
        return 0.07*np.exp(-V/20.)

    def beta_h(V):
        return 1/(np.exp((30 - V)/10.) + 1)

    def h_f(h, V):
        return alpha_h(V)*(1 - h) - beta_h(V)*h

    def h_inf(V):
        return alpha_h(V)/(alpha_h(V) + beta_h(V))


    def dXdt(X, t):
        V, h, m, n = X

        g_Na = gbar_Na*(m**3)*h
        g_K = gbar_K*(n**4)
        g_l = gbar_L

        dmdt = m_f(m, V)
        dhdt = h_f(h, V)
        dndt = n_f(n, V)

        dVdt = (I(t) - g_Na*(V - E_Na) - g_K*(V - E_K) - g_l*(V - E_l))/C_m

        return [dVdt, dhdt, dmdt, dndt]


    initial_conditions = [V_0, h_0, m_0, n_0]

    X = odeint(dXdt, initial_conditions, time)
    values = X[:, 0]

    # Only return from 5 seconds onwards, as in the Valderrama paper
    values = values[time > 5]
    time = time[time > 5]

    # Add info needed by certain spiking features and efel features
    info = {"stimulus_start": time[0], "stimulus_end": time[-1]}

    return time, values, info