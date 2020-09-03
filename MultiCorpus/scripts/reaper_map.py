from ftis.analyser import (
    FluidOnsetslice, 
    FluidMFCC, 
    Stats, 
    Standardise,
    UMAP,
    ExplodeAudio
)
from ftis.process import FTISProcess as Chain
from ftis.corpus import Corpus, PathLoader


analysis = Chain(
    source = Corpus("../../reaper/highgain_source/bounces"), 
    folder = "../outputs/em_detailed_segmentation"
)

analysis.add(
    # Segmentation
    FluidOnsetslice(framedelta=20, minslicelength=2, filtersize=5, threshold=0.3, metric=0, cache=1),
    ExplodeAudio(),
    # Analysis
    FluidMFCC(discard=True, numcoeffs=20, fftsettings=[4096, 512, 4096]),
    Stats(numderivs=1),
    Standardise(),
    UMAP(components=2)
)

if __name__ == "__main__":
    analysis.run()
