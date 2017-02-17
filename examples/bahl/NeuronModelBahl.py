from uncertainpy import NeuronModel

class NeuronModelBahl(NeuronModel):
    def __init__(self, parameters=None, adaptive_model=False):
        NeuronModel.__init__(self, parameters=parameters,
                             adaptive_model=adaptive_model)

        self.reset_properties()

        self.adaptive_model = True

        self.model_path = "bahl_neuron_model"
        self.model_file = "mosinit.hoc"



    def setParameterValues(self, parameters):
        for parameter in parameters:
            self.h(parameter + " = " + str(parameters[parameter]))

        self.h("recalculate_passive_properties()")
        self.h("recalculate_channel_densities()")
