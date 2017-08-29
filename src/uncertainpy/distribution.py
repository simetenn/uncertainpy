import chaospy as cp


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
