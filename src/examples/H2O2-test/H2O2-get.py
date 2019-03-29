import pickle
#import numpy as np

with open("H2O2-bak.dat", "rb") as f:
    d = pickle.load(f)

print(d["E_0"])
print(d["E_1"])
print(d["E_2"])

