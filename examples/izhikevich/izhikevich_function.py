import numpy as np


def izhikevich(a=0.02, b=0.2, c=-50, d=2):
    time_end = 100
    dt = 0.25
    v0 = -70

    time = np.linspace(0, time_end, time_end/dt)

    def I(time):
        if time >= 10:
            return 10
        else:
            return 0

    info = {"stimulus_start": 10,
            "stimulus_end": time_end}

    def f(u_in, time):
        v, u = u_in

        dvdt = 0.04*v**2 + 5*v + 140 - u + I(time)
        dudt = a*(b*v - u)

        return np.array([dvdt, dudt])

    u = np.zeros((len(time), 2))
    u[0, 0] = v0
    u[0, 1] = b*v0

    values = [v0]*len(time)

    # Runge Kutta 4
    dt2 = dt/2.0
    for n in range(len(time) - 1):
        K1 = dt*f(u[n], time[n])
        K2 = dt*f(u[n] + 0.5*K1, time[n] + dt2)
        K3 = dt*f(u[n] + 0.5*K2, time[n] + dt2)
        K4 = dt*f(u[n] + K3, time[n] + dt)
        u_new = u[n] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)

        if u_new[0] > 30:
            u_new[0] = c
            u_new[1] += d
            values[n] = 30
        else:
            values[n] = u_new[0]

        u[n+1] = u_new

    return time, np.array(values), info
