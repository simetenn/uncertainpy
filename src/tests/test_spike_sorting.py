from IzhikevichModel import IzhikevichModel


model = IzhikevichModel()

# from NeuronModel import NeuronModel
#
# modelfile = "INmodel.hoc"
# modelpath = "/home/simen/Dropbox/phd/parameter_estimation/uncertainpy/models/neuron_models/dLGN_modelDB/"
# model = NeuronModel(modelfile, modelpath)
# model.load()

t, U = model.run()


import pylab as plt
plt.plot(t, U)
plt.show()

from uncertainpy import Spikes

spikes = Spikes()
spikes.detectSpikes(t, U)
spikes.plot("test.png")


# FS = float(len(t))/(t[-1] - t[0])*1000
# U_min = abs(U.min())
#
# U += U_min
#
# data_dict = {"data": np.array([U]), 'n_contacts': 1, "FS": FS}
# # plt.plot(t, U)
# # plt.show()
#
#
# def spike_times(t1, U1):
#     print U1.min()
#     thresh = 8*np.sqrt(U1.var())
#     print thresh
#
# spike_times(t, U)
#
# spike_times = extract.detect_spikes(data_dict)
# waves_dict = extract.extract_spikes(data_dict, spike_times, [-1, 1], resample=1)
#
# print waves_dict["data"].shape
# feature = features.fetSpTime(waves_dict)
#
# print feature
# #
# # plt.plot(waves_dict["data"][:, 0, 0])
# # plt.show()
#
# nr_spikes = len(spike_times["data"])
#
# print nr_spikes
# spikes = []
# for i in xrange(nr_spikes):
#     # time =
#     spikes.append((time, spike))
#
# print spike_times
# print waves_dict
# U -= U_min
# #
# plt.plot(t, U)
# plt.show()

# import numpy as np
# data = np.array([[0, 1, 0, 0, 0, 1]])
# print data.shape
#
# raw_dict = {'data': np.array([[0, 1, 0, 0, 0, 1]]),
#             'FS': 10000,
#             'n_contacts': 1}
# from spike_sort.core.extract import detect_spikes
# spt_dict = detect_spikes(raw_dict, thresh=0.8)
# print(spt_dict.keys())
# print('Spike times (ms): {0}'.format(spt_dict['data']))
