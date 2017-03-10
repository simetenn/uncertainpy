import chaospy as cp

# __all__ = ["Distribution"]
# __version__ = "0.1"

class Distribution():
    def __init__(self, interval, function=None):
        self.interval = interval
        self.function = function

    def __call__(self, parameter):
        return self.function(parameter, self.interval)

    def normal(self, parameter):
        if parameter == 0:
            raise ValueError("Creating a percentage distribution around 0 does not work")

        return cp.Normal(parameter, abs(self.interval*parameter))

    def uniform(self, parameter):
        if parameter == 0:
            raise ValueError("Creating a percentage distribution around 0 does not work")

        return cp.Uniform(parameter - abs(self.interval/2.*parameter),
                          parameter + abs(self.interval/2.*parameter))


# 
# def create_distribution(interval, func):
#     def distribution(parameter):
#         if parameter == 0:
#             raise ValueError("Creating a percentage distribution around 0 does not work")
#
#         return func
#     return distribution
#

def uniform(interval):
    def distribution(parameter):
        if parameter == 0:
            raise ValueError("Creating a percentage distribution around 0 does not work")

        return cp.Uniform(parameter - abs(interval/2.*parameter),
                          parameter + abs(interval/2.*parameter))
    return distribution

def normal(interval):
    def distribution(parameter):
        if parameter == 0:
            raise ValueError("Creating a percentage distribution around 0 does not work")

        return cp.Normal(parameter, abs(interval*parameter))
    return distribution
