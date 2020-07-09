# Description
"""
A heavy sub-segmentation of the classified samples from the source.
The idea is to return short transients and micro gestures, wittling down to the micro level as much as is perceptually reasonable.
"""

from ftis.analyser import ClusteredSegmentation, FluidOnsetslice
from ftis.process import FTISProcess as Chain

src = "outputs/classification/4_Split/0"
folder = "outputs/micro_segmentation"

process = Chain(
    source=src, 
    folder=folder
)

initial_segmentation = FluidOnsetslice(threshold=0.3)
cluster_segmentation = ClusteredSegmentation()

process.add(
    initial_segmentation,
    cluster_segmentation
)

if __name__ == "__main__":
    process.run()
