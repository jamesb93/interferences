# Description
"""
Analyse each item belonging to a cluster so that they can be ranked in order of a descriptor.
"""

from ftis.analyser import Flux
from ftis.process import FTISProcess as Chain
from ftis.common.io import read_json
import numpy as np

src = "outputs/segments/2_ExplodeAudio"
folder = "outputs/metacluster_analysis"

process = Chain(
    source=src, 
    folder=folder
)

flux = Flux(cache=True)
process.add(
    flux
)

if __name__ == "__main__":
    process.run()
    # print(flux.output)
    clusters = read_json("outputs/classification/3_AGCluster")
    for k in clusters.keys():
        buf = []
        for v in clusters[k]:
            for point in flux.output[v]:
                buf.append(point)
        print(f"Cluster {k}: {np.median(buf)}")
            
        


