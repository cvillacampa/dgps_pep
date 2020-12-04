import os

import numpy as np
import psutil


def calculate_ETA(last_epoch_times, current_iteration, max_iterations):
    """Calculates remaining training time in seconds

    Args:
        last_epoch_times (list/deque): Running time of last epochs
        current_iteration (int): current iteration number
        max_iterations (int): Max training iterations
    """
    mean_iteration_time = np.mean(last_epoch_times)
    remaining_iterations = max_iterations - current_iteration
    return mean_iteration_time * remaining_iterations


def calculate_ETA_str(last_epoch_times, current_iteration, max_iterations):
    """Calculates remaining training time, returning a string

    Args:
        last_epoch_times (list/deque): Running time of last epochs
        current_iteration (int): current iteration number
        max_iterations (int): Max training iterations
    """
    if current_iteration < 5:
        return "-"

    eta = calculate_ETA(last_epoch_times, current_iteration, max_iterations)
    sec_to_min = 60
    sec_to_hour = 3600
    sec_to_day = 86400
    if eta < sec_to_min:
        return "{:1.0f}s".format(eta)
    if eta < sec_to_hour:
        return "{:1.0f}min, {:1.0f}s".format(eta // sec_to_min, (eta % sec_to_min))
    if eta < sec_to_day:
        return "{:1.0f}h, {:1.0f}min".format(
            eta // sec_to_hour, (eta % sec_to_hour) // sec_to_min
        )

    return "{:1.0f}day, {:1.0f}h".format(
        eta // sec_to_day, (eta % sec_to_day) // sec_to_hour
    )


def extend_dimension_if_1d(np_array):
    return np_array[:, None] if np_array.ndim == 1 else np_array


def memory_used():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_usage = py.memory_info()[0] / 2.0 ** 30  # memory use in GB
    return memory_usage

