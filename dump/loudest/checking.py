from ftis.common.io import read_json

f = read_json("2.0-MFCC.Stats.json")
p = 0
for k, v in f.items():
    if len(v) != p:
        print(k)
    p = len(v)
