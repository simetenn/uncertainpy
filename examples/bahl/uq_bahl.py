import uncertainpy as un
import chaospy as cp

from .bahl import NeuronModelBahl

parameter_list = [["e_pas", -80, cp.Uniform(-60, -85)],
                 ["apical Ra", 261, cp.Uniform(150, 300)]]

model = NeuronModelBahl(stimulus_start=100, stimulus_end=600)


features = un.SpikingFeatures()

UQ = un.UncertaintyQuantification(model=model,
                                  parameters=parameter_list,
                                  features=features,
                                  plot=True)

UQ.quantify(plot_condensed=False,
            plot_model=True)
UQ.quantify(single=True,
            plot_model=False)
