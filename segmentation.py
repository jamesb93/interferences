from ftis.analyser import (
    FluidNoveltyslice,
    ClusteredSegmentation,
    ExplodeAudio
)
from ftis.process import FTISProcess as Chain

src = "reaper/highgain_source/bounces"
folder = "segments2"

process = Chain(
    source=src, 
    folder=folder
)

process.add(
    FluidNoveltyslice(threshold=0.4,minslicelength=4096, feature=0),
    ClusteredSegmentation(numclusters=3, windowsize=5),
    ExplodeAudio()
)

process.run()
