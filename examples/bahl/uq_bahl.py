import uncertainpy as un
import chaospy as cp

# Subclassing NeuronModel
class NeuronModelBahl(un.NeuronModel):
    def __init__(self, stimulus_start=None, stimulus_end=None):
        # Hardcode the path of the Bahl neuron model
        super(NeuronModelBahl, self).__init__(adaptive=True,
                                              path="bahl_neuron_model",
                                              stimulus_start=stimulus_start,
                                              stimulus_end=stimulus_end)

    # Reimplement the set_parameters method used by run
    def set_parameters(self, parameters):
        for parameter in parameters:
            self.h(parameter + " = " + str(parameters[parameter]))

        # These commands must be added for this specific
        # model to recalculate the parameters after they have been set
        self.h("recalculate_passive_properties()")
        self.h("recalculate_channel_densities()")


# Initialize the model with the start and end time of the stimulus
model = NeuronModelBahl(stimulus_start=100, stimulus_end=600)

# Define a parameter list and use it directly
parameters = {"e_pas": -80, cp.Uniform(-60, -85),
              "apical Ra": 261, cp.Uniform(150, 300)}

# Initialize the features
features = un.SpikingFeatures()

# Perform the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model,
                                  parameters=parameters,
                                  features=features)
UQ.quantify()