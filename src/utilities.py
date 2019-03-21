import numpy as np


def val_from_fchk(key, file_path):
    flag_read = False
    expect_size = -1
    vec = []
    with open(file_path, "r") as file:
        for l in file:
            if (l[:len(key)] == key):
                try:
                    expect_size = int(l[len(key):].split()[2])
                    flag_read = True
                    continue
                except IndexError:
                    try:
                        return float(l[len(key):].split()[1])
                    except IndexError:
                        continue
            if (flag_read):
                try:
                    vec += [ float(i) for i in l.split() ]
                except ValueError:
                    break
    if len(vec) != expect_size:
        raise ValueError("Number of expected size is not consistent with read-in size!")
    return np.array(vec)
