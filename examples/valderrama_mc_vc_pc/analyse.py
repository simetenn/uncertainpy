import glob
from tqdm import tqdm

import uncertainpy as un
import numpy as np
import h5py



def calculate_error(glob_pattern, correct_data, base="data/"):
    files = glob.glob(base + glob_pattern)

    mean_errors = {}
    variance_errors = {}
    for file in tqdm(files):
        data = un.Data(file)

        mean = data["valderrama"].mean
        correct_mean = correct_data["valderrama"].mean
        variance = data["valderrama"].variance
        correct_variance = correct_data["valderrama"].variance

        dt = data["valderrama"].time[1] - data["valderrama"].time[0]
        nr_evaluations = len(data["valderrama"].evaluations)

        mean_error = dt*np.sum(np.abs((correct_mean - mean)/correct_mean))
        variance_error = dt*np.sum(np.abs((correct_variance - variance)/correct_variance))

        if nr_evaluations not in mean_errors:
            mean_errors[nr_evaluations] = [mean_error]
        else:
            mean_errors[nr_evaluations].append(mean_error)

        if nr_evaluations not in variance_errors:
            variance_errors[nr_evaluations] = [variance_error]
        else:
            variance_errors[nr_evaluations].append(variance_error)

        del data

    sorted_nr_evaluations = []
    average_mean_errors = []
    average_variance_errors = []

    for evaluation in sorted(mean_errors.keys()):
        sorted_nr_evaluations.append(evaluation)
        average_mean_errors.append(np.mean(mean_errors[evaluation]))
        average_variance_errors.append(np.mean(variance_errors[evaluation]))

    return sorted_nr_evaluations, average_mean_errors, average_variance_errors




# 3 uncertain parameters
correct_data_3 =  un.Data("data/parameters_3/correct.h5")

pc_evaluations_3, pc_mean_errors_3, pc_variance_errors_3 = calculate_error("parameters_3/pc_*",  correct_data_3)
mc_evaluations_3, mc_mean_errors_3, mc_variance_errors_3 = calculate_error("parameters_3/mc_*",  correct_data_3)


# 11 uncertain parameters
correct_data_11 =  un.Data("data/parameters_11/correct.h5")

pc_evaluations_11, pc_mean_errors_11, pc_variance_errors_11 = calculate_error("parameters_11/pc_*", correct_data_11)
mc_evaluations_11, mc_mean_errors_11, mc_variance_errors_11 = calculate_error("parameters_11/mc_*", correct_data_11)


with h5py.File("analysed_data.h5", "w") as f:
    f.create_dataset("pc_evaluations_3", data=pc_evaluations_3)
    f.create_dataset("pc_mean_errors_3", data=pc_mean_errors_3)
    f.create_dataset("pc_variance_errors_3", data=pc_variance_errors_3)

    f.create_dataset("mc_evaluations_3", data=mc_evaluations_3)
    f.create_dataset("mc_mean_errors_3", data=mc_mean_errors_3)
    f.create_dataset("mc_variance_errors_3", data=mc_variance_errors_3)

    f.create_dataset("pc_evaluations_11", data=pc_evaluations_11)
    f.create_dataset("pc_mean_errors_11", data=pc_mean_errors_11)
    f.create_dataset("pc_variance_errors_11", data=pc_variance_errors_11)

    f.create_dataset("mc_evaluations_11", data=mc_evaluations_11)
    f.create_dataset("mc_mean_errors_11", data=mc_mean_errors_11)
    f.create_dataset("mc_variance_errors_11", data=mc_variance_errors_11)

