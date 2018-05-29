import glob
from tqdm import tqdm

import uncertainpy as un
import numpy as np
import h5py



def calculate_error(glob_pattern, exact_data, base="data/"):
    files = glob.glob(base + glob_pattern)

    exact_mean = exact_data["valderrama"].mean
    exact_variance = exact_data["valderrama"].variance
    exact_sobol = exact_data["valderrama"].sobol_first

    mean_errors = {}
    variance_errors = {}
    sobol_errors = {}
    for file in tqdm(files):
        data = un.Data(file)

        mean = data["valderrama"].mean
        variance = data["valderrama"].variance
        sobol = data["valderrama"].sobol_first

        dt = data["valderrama"].time[1] - data["valderrama"].time[0]
        nr_evaluations = data["valderrama"].evaluations[0]
        sobol_evaluations = data["valderrama"].evaluations[1]

        mean_error = dt*np.sum(np.abs((exact_mean - mean)/exact_mean))
        variance_error = dt*np.sum(np.abs((exact_variance - variance)/exact_variance))
        sobol_error = dt*np.sum(np.abs((exact_sobol - sobol)/exact_sobol), axis=1)
        sobol_error = np.mean(sobol_error)

        if nr_evaluations not in mean_errors:
            mean_errors[nr_evaluations] = [mean_error]
        else:
            mean_errors[nr_evaluations].append(mean_error)

        if nr_evaluations not in variance_errors:
            variance_errors[nr_evaluations] = [variance_error]
        else:
            variance_errors[nr_evaluations].append(variance_error)

        if sobol_evaluations not in sobol_errors:
            sobol_errors[sobol_evaluations] = [sobol_error]
        else:
            sobol_errors[sobol_evaluations].append(sobol_error)

        del data


    sorted_nr_evaluations = []
    average_mean_errors = []
    average_variance_errors = []
    for evaluation in sorted(mean_errors.keys()):
        sorted_nr_evaluations.append(evaluation)
        average_mean_errors.append(np.mean(mean_errors[evaluation]))
        average_variance_errors.append(np.mean(variance_errors[evaluation]))


    sorted_sobol_evaluations = []
    average_sobol_errors = []
    for evaluation in sorted(sobol_errors.keys()):
        sorted_sobol_evaluations.append(evaluation)
        average_sobol_errors.append(np.mean(sobol_errors[evaluation]))


    return sorted_nr_evaluations, average_mean_errors, average_variance_errors, sorted_sobol_evaluations, average_sobol_errors




# 3 uncertain parameters
exact_data_3 =  un.Data("data/parameters_3/exact.h5")

pc_evaluations_3, pc_mean_errors_3, pc_variance_errors_3, pc_sobol_evaluations_3, pc_sobol_errors_3 = calculate_error("parameters_3/pc_*",  exact_data_3)
mc_evaluations_3, mc_mean_errors_3, mc_variance_errors_3, mc_sobol_evaluations_3, mc_sobol_errors_3 = calculate_error("parameters_3/mc_*",  exact_data_3)


# 11 uncertain parameters
exact_data_11 =  un.Data("data/parameters_11/exact.h5")

pc_evaluations_11, pc_mean_errors_11, pc_variance_errors_11, pc_sobol_evaluations_11, pc_sobol_errors_11 = calculate_error("parameters_11/pc_*", exact_data_11)
mc_evaluations_11, mc_mean_errors_11, mc_variance_errors_11, mc_sobol_evaluations_11, mc_sobol_errors_11 = calculate_error("parameters_11/mc_*", exact_data_11)


with h5py.File("analysed_data.h5", "w") as f:
    f.create_dataset("pc_evaluations_3", data=pc_evaluations_3)
    f.create_dataset("pc_mean_errors_3", data=pc_mean_errors_3)
    f.create_dataset("pc_variance_errors_3", data=pc_variance_errors_3)
    f.create_dataset("pc_sobol_evaluations_3", data=pc_sobol_evaluations_3)
    f.create_dataset("pc_sobol_errors_3", data=pc_sobol_errors_3)

    f.create_dataset("mc_evaluations_3", data=mc_evaluations_3)
    f.create_dataset("mc_mean_errors_3", data=mc_mean_errors_3)
    f.create_dataset("mc_variance_errors_3", data=mc_variance_errors_3)
    f.create_dataset("mc_sobol_evaluations_3", data=mc_sobol_evaluations_3)
    f.create_dataset("mc_sobol_errors_3", data=mc_sobol_errors_3)

    f.create_dataset("pc_evaluations_11", data=pc_evaluations_11)
    f.create_dataset("pc_mean_errors_11", data=pc_mean_errors_11)
    f.create_dataset("pc_variance_errors_11", data=pc_variance_errors_11)
    f.create_dataset("pc_sobol_evaluations_11", data=pc_sobol_evaluations_11)
    f.create_dataset("pc_sobol_errors_11", data=pc_sobol_errors_11)


    f.create_dataset("mc_evaluations_11", data=mc_evaluations_11)
    f.create_dataset("mc_mean_errors_11", data=mc_mean_errors_11)
    f.create_dataset("mc_variance_errors_11", data=mc_variance_errors_11)
    f.create_dataset("mc_sobol_evaluations_11", data=mc_sobol_evaluations_11)
    f.create_dataset("mc_sobol_errors_11", data=mc_sobol_errors_11)

