# Description
"""
Arrange segmentations along a single axis using dimension reduction
"""

from ftis.analyser import FluidMFCC, UMAP, Stats, Normalise
from ftis.process import FTISProcess as Chain
from ftis.common.conversion import samps2ms

src = "outputs/micro_segmentation/2_ExplodeAudio/"
folder = "outputs/oned"

process = Chain(
    source=src, 
    folder=folder
)

mfcc = FluidMFCC(cache=True)
stats = Stats(flatten=True, numderivs=1, cache=True)
umap = UMAP(components=1, cache=True)
normalise = Normalise()

process.add(
    mfcc,
    stats,
    umap,
    normalise)

if __name__ == "__main__":
    process.run()