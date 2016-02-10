from uncertainpy.models import NeuronModel

import sys
print sys.modules.keys()


modelfile = "INmodel.hoc"
modelpath = "/home/simen/Dropbox/phd/parameter_estimation/uncertainpy/models/neuron_models/dLGN_modelDB/"
model = NeuronModel(modelfile, modelpath)
model.load()

#t, U = model.run()
