try:
    import nest

    prerequisites = True
except ImportError:
    prerequisites = False

import numpy as np

from .model import Model


class NestModel(Model):
    def __init__(self,
                 run_function=None,
                 adaptive=False,
                 labels=["time [ms]", "Neuron nr", "Spiking probability"]):


        if not prerequisites:
            raise ImportError("Nest model requires: nest")


        super(NestModel, self).__init__(run_function=run_function,
                                        adaptive=adaptive,
                                        labels=labels)



    def postprocess(self, t_stop, U):
        dt = nest.GetKernelStatus()["resolution"]
        t = np.arange(0, t_stop, dt)

        expanded_spiketrains = []
        for spiketrain in U:
            binary_spike = np.zeros(len(t))
            binary_spike[np.in1d(t, spiketrain)] = 1

            expanded_spiketrains.append(binary_spike)

        U = np.array(expanded_spiketrains)

        return t, U