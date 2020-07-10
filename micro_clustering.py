# Description
"""
A heavy sub-segmentation of the classified samples from the source.
The idea is to return short transients and micro gestures, wittling down to the micro level as much as is perceptually reasonable.
"""

from ftis.analyser import FluidMFCC, UmapDR, HDBSCluster, Stats
from ftis.process import FTISProcess as Chain
from ftis.common.conversion import samps2ms

src = "outputs/micro_segmentation/2_ExplodeAudio/"
folder = "outputs/micro_clustering"

process = Chain(
    source=src, 
    folder=folder
)

mfcc = FluidMFCC(cache=True)
stats = Stats(flatten=True, numderivs=1)
umap = UmapDR(components=27)
cluster = HDBSCluster()

process.add(
    mfcc,
    stats,
    umap,
    cluster
)

if __name__ == "__main__":
    process.run()