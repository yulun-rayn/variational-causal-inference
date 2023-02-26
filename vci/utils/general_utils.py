import os
import json
import logging

import numpy as np

def initialize_logger(artifact_path, name=None, level='INFO'):
    logfile = os.path.join(artifact_path, 'log.txt')
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)

    handler_console = logging.StreamHandler()
    handler_file    = logging.FileHandler(logfile)

    logger.addHandler(handler_console)
    logger.addHandler(handler_file)
    return logger

def pjson(s):
    print(json.dumps(s), flush=True)

def ljson(s):
    logging.info(json.dumps(s))

def unique_ind(records_array):
    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(records_array)

    # sorts records array so all unique elements are together 
    sorted_records_array = records_array[idx_sort]

    # returns the unique values, the index of the first occurrence of a value
    vals, idx_start = np.unique(sorted_records_array, return_index=True)

    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])

    return dict(zip(vals, res))
