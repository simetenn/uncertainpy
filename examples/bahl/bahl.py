from uncertainpy import NeuronModel

class NeuronModelBahl(NeuronModel):
    def __init__(self):
        NeuronModel.__init__(self,
                             adaptive=True,
                             model_file="mosinit.hoc",
                             model_path="bahl_neuron_model")


    def set_parameters(self, parameters):
        for parameter in parameters:
            self.h(parameter + " = " + str(parameters[parameter]))

        # These commands must be added for this specific
        # model to set the parameters
        # self.h("recalculate_passive_properties()")
        # self.h("recalculate_channel_densities()")
