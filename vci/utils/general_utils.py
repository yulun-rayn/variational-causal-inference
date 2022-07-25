import json
import numpy as np

def pjson(s):
    """
    Prints a string in JSON format and flushes stdout
    """
    print(json.dumps(s), flush=True)

def sjson(s, f):
    # create json object from dictionary
    s = json.dumps(s)

    # open file for writing, "w" 
    f = open(f, "w")

    # write json object to file
    f.write(s)

    # close file
    f.close()

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
