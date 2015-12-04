import chaospy as cp

__all__ = ["Distribution"]
__version__ = "0.1"

class Distribution():
    def __init__(self, interval, function=None):
        self.interval = interval
        self.function = function

    def __call__(self, parameter):
        return self.function(parameter, self.interval)

    def normal(self, parameter):
        return cp.Normal(parameter, abs(self.interval*parameter))

    def uniform(self, parameter):
        return cp.Uniform(parameter - abs(self.interval/2.*parameter),
                          parameter + abs(self.interval/2.*parameter))
