from uncertainpy import NeuronModel

class NeuronModelBahl(NeuronModel):
    def __init__(self, stimulus_start=None, stimulus_end=None):
        NeuronModel.__init__(self,
                             adaptive=True,
                             file="mosinit.hoc",
                             path="bahl_neuron_model",
                             stimulus_start=stimulus_start,
                             stimulus_end=stimulus_start)


    def set_parameters(self, parameters):
        for parameter in parameters:
            self.h(parameter + " = " + str(parameters[parameter]))

        # These commands must be added for this specific
        # model to set the updated the parameters after they have been set
        self.h("recalculate_passive_properties()")
        self.h("recalculate_channel_densities()")
