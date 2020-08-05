from flucoma import fluid
from flucoma.utils import get_buffer, cleanup
from pathlib import Path
from ftis.common.io import write_json, read_json
from statistics import stdev

def normalise(arr):
    mi = min(arr)
    ma = max(arr)
    ra = abs(ma-mi)
    try:
        return [(x-mi) / ra for x in arr]
    except:
        print(f"FUCKED UP{arr}")

def mean(arr):
    return sum(arr) / len(arr)

def deriv(arr):
    return [y - x for x, y in zip(arr, arr[1:])]


# files = [x for x in Path("outputs/micro_segmentation/2_ExplodeAudio").iterdir()]
# clusters = read_json("outputs/micro_clustering/5_HDBSCluster.json")
# files = [x for x in clusters["37"]]

files = [x for x in Path("outputs/concat").iterdir() if x.suffix == ".wav"]
print(files)

d = {}

for i, f in enumerate(files):
    print(f)
    print(i / len(files))
    ts = get_buffer(fluid.transientslice(f))
    
    if ts[0] != 0:
        ts.insert(0, 0)

    if len(ts) <= 2 and ts[0] == 0.0:
        d[str(f)] = -1
    else:
        # Let's grab the orderedness of the onsets
        norm = normalise(ts)
        average = mean(norm)
        robustified = [x / average for x in norm]
        first_deriv = deriv(robustified)
        d[str(f)] = stdev(first_deriv)

mi = 99999
ma = -99999
for v in d.values():
    if v > ma:
        ma = v
    if v < mi and v != -1:
        mi = v
ra = abs(ma-mi)
for k, v in zip(d.keys(), d.values()):
    d[k] = (v-mi) / ra

write_json("evenness.json", d)
cleanup()
    