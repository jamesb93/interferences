from ftis.common.io import read_json
import numpy as np
from shutil import copyfile
from pathlib import Path

f = read_json("1_Stats")
values = []

for x in f.values():
    values.append(x[12])

values = np.array(values)
b = np.percentile(values, 20)
print(b)

for k, v in zip(f.keys(), f.values()):
    if v[12] <= b:
        p = Path(k)
        copyfile(
            p.resolve(), 
            f"../bottom_ten_percent/{p.name}")