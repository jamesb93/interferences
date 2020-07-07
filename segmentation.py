# Description
"""
Segment using clustered segmentation approach and create an audio file per segment
"""

from ftis.analyser import (
    FluidNoveltyslice,
    ClusteredSegmentation,
    ExplodeAudio
)
from ftis.process import FTISProcess as Chain

src = "reaper/highgain_source/bounces"
folder = "outputs/segments"

process = Chain(
    source=src, 
    folder=folder
)

process.add(
    FluidNoveltyslice(threshold=0.1,minslicelength=4096, feature=0, cache=True),
    ClusteredSegmentation(numclusters=3, windowsize=5, cache=True),
    ExplodeAudio()
)

if __name__ == "__main__":
    process.run()
