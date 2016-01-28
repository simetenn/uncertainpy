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

        # original_parameters, uncertain_parameters, distributions,

        self.t_start = time.time()

        self.output_dir_data = output_dir_data
        self.output_dir_figures = output_dir_figures


        # if os.path.isdir(output_dir_data):
        #     shutil.rmtree(output_dir_data)
        # os.makedirs(output_dir_data)

        if not os.path.isdir(output_dir_data):
            os.makedirs(output_dir_data)


        self.uncertainty_estimation = UncertaintyEstimation(model,
                                                       feature_list=feature_list,
                                                       features=features,
                                                       figureformat=figureformat,
                                                       output_dir_data=self.output_dir_data,
                                                       output_data_filename=model.__class__.__name__,
                                                       supress_model_graphics=supress_model_graphics,
                                                       supress_model_output=supress_model_output,
                                                       CPUs=CPUs,
                                                       interpolate_union=interpolate_union,
                                                       rosenblatt=rosenblatt,
                                                       nr_mc_samples=nr_mc_samples,
                                                       **kwargs)

    def exploreParameters(self, distributions):
        for distribution_function in distributions:
            for interval in distributions[distribution_function]:
                print "Running for: " + distribution_function + " " + str(interval)

                distribution = getattr(Distribution(interval), distribution_function)
                self.uncertainty_estimation.model.setAllDistributions(distribution)

                tmp_output_dir_data = os.path.join(self.output_dir_data,
                                                   distribution_function + "_%g" % interval)
                current_output_dir_figures = os.path.join(self.output_dir_figures,
                                                          distribution_function + "_%g" % interval)

                self.uncertainty_estimation.output_dir_figures = current_output_dir_figures
                self.uncertainty_estimation.output_dir_data = tmp_output_dir_data


                self.uncertainty_estimation.singleParameters()
                self.uncertainty_estimation.allParameters()



    def compareMC(self, nr_mc_samples):
        run_times = []
        self.uncertainty_estimation.output_dir_figures = os.path.join(self.output_dir_figures, "pc_compare")
        self.uncertainty_estimation.output_dir_data = os.path.join(self.output_dir_data, "pc-compare")

        time_1 = time.time()
        self.uncertainty_estimation.allParameters()
        run_times.append(time.time() - time1)


        for nr_mc_sample in nr_mc_samples:
            print "Running for: " + str(nr_mc_sample)

            current_output_dir_figures = os.path.join(self.output_dir_figures,
                                                      "mc-compare_" + str(nr_mc_sample))
            tmp_output_dir_data = os.path.join(self.output_dir_data,
                                               "mc_-compare_" + str(nr_mc_sample))

            self.uncertainty_estimation.output_dir_figures = current_output_dir_figures
            self.uncertainty_estimation.output_dir_data = tmp_output_dir_data

            time_1 = time.time()
            self.uncertainty_estimation.allParametersMC()
            run_times.append(time.time() - time1)

        return run_times



    def timePassed(self):
        return time.time() - self.t_start
