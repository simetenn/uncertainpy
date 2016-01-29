import time
import os
import shutil
import h5py
import sys

import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
import multiprocessing as mp

from uncertainty import UncertaintyEstimation
from distribution import Distribution

class UncertaintyEstimations():
    def __init__(self, model,
                 feature_list=[],
                 features=None,
                 output_dir_figures="figures/",
                 figureformat=".png",
                 output_dir_data="data/",
                 supress_model_graphics=True,
                 supress_model_output=True,
                 CPUs=mp.cpu_count(),
                 interpolate_union=False,
                 rosenblatt=False,
                 nr_mc_samples=10**3,
                 **kwargs):
        """
        Options can also be sent to the feature
        kwargs:
        feature_options = {keyword1: value1, keyword2: value2}
        """

        # Figures are always saved on the format:
        # output_dir_figures/distribution_interval/parameter_value-that-is-plotted.figure-format

        self.uncertainty_estimations = {}

        # original_parameters, uncertain_parameters, distributions,

        self.model = model

        self.output_dir_figures = output_dir_figures
        self.output_dir_data = output_dir_data

        self.supress_model_graphics = supress_model_graphics
        self.supress_model_output = supress_model_output
        self.CPUs = CPUs
        self.interpolate_union = interpolate_union
        self.rosenblatt = rosenblatt
        self.figureformat = figureformat
        self.features = features
        self.feature_list = feature_list
        self.nr_mc_samples = nr_mc_samples

        self.kwargs = kwargs

        self.t_start = time.time()

        if not os.path.isdir(output_dir_data):
            os.makedirs(output_dir_data)


    def exploreParameters(self, distributions):
        for distribution_function in distributions:
            for interval in distributions[distribution_function]:
                current_output_dir_figures = os.path.join(self.output_dir_figures,
                                                          distribution_function + "_%g" % interval)
                distribution = getattr(Distribution(interval), distribution_function)

                self.model.setAllDistributions(distribution)

                name = distribution_function + "_" + str(interval)
                print "Running for: " + distribution_function + " " + str(interval)

                tmp_output_dir_data = os.path.join(self.output_dir_data, distribution_function + "_%g" % interval)

                self.uncertainty_estimations[name] = UncertaintyEstimation(self.model,
                                                               feature_list=self.feature_list,
                                                               features=self.features,
                                                               output_dir_figures=current_output_dir_figures,
                                                               figureformat=self.figureformat,
                                                               output_dir_data=tmp_output_dir_data,
                                                               output_data_filename=self.model.__class__.__name__,
                                                               supress_model_graphics=self.supress_model_graphics,
                                                               supress_model_output=self.supress_model_output,
                                                               CPUs=self.CPUs,
                                                               interpolate_union=self.interpolate_union,
                                                               rosenblatt=self.rosenblatt,
                                                               nr_mc_samples=self.nr_mc_samples,
                                                               **self.kwargs)

                self.uncertainty_estimations[name].singleParameters()
                self.uncertainty_estimations[name].allParameters()



    def compareMC(self, nr_mc_samples):
        run_times = []

        name = "pc-compare"
        output_dir_figures = os.path.join(self.output_dir_figures, name)
        output_dir_data = os.path.join(self.output_dir_data, name)

        self.uncertainty_estimations[name] = UncertaintyEstimation(self.model,
                                                       feature_list=self.feature_list,
                                                       features=self.features,
                                                       output_dir_figures=output_dir_figures,
                                                       figureformat=self.figureformat,
                                                       output_dir_data=output_dir_data,
                                                       output_data_filename=self.model.__class__.__name__,
                                                       supress_model_graphics=self.supress_model_graphics,
                                                       supress_model_output=self.supress_model_output,
                                                       CPUs=self.CPUs,
                                                       interpolate_union=self.interpolate_union,
                                                       rosenblatt=self.rosenblatt,
                                                       nr_mc_samples=nr_mc_samples,
                                                       **self.kwargs)

        time_1 = time.time()
        #self.uncertainty_estimations[name].allParameters()
        run_times.append(time.time() - time_1)


        for nr_mc_sample in nr_mc_samples:
            print "Running for: " + str(nr_mc_sample)


            name = "mc-compare_" + str(nr_mc_sample)
            current_output_dir_figures = os.path.join(self.output_dir_figures, name)
            tmp_output_dir_data = os.path.join(self.output_dir_data, name)

            self.uncertainty_estimations[name] = UncertaintyEstimation(self.model,
                                                               feature_list=self.feature_list,
                                                               features=self.features,
                                                               output_dir_figures=current_output_dir_figures,
                                                               figureformat=self.figureformat,
                                                               output_dir_data=tmp_output_dir_data,
                                                               output_data_filename=self.model.__class__.__name__,
                                                               supress_model_graphics=self.supress_model_graphics,
                                                               supress_model_output=self.supress_model_output,
                                                               CPUs=self.CPUs,
                                                               interpolate_union=self.interpolate_union,
                                                               rosenblatt=self.rosenblatt,
                                                               **self.kwargs)

            time_1 = time.time()
            self.uncertainty_estimations[name].allParametersMC()
            run_times.append(time.time() - time_1)

        return run_times



    def timePassed(self):
        return time.time() - self.t_start
