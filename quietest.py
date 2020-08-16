from ftis.analyser import FluidMFCC, Stats, Standardise, UmapDR, Normalise, HDBSCluster
from ftis.corpus import CorpusLoader, CorpusFilter
from ftis.process import FTISProcess as Chain

src = "outputs/micro_segmentation/2_ExplodeAudio"
folder = "outputs/quitest"

process = Chain(
    source=src, 
    folder=folder
)

process.add(
    CorpusLoader(cache=1),
    CorpusFilter(max_loudness=10, cache=1),
    FluidMFCC(discard=True, cache=1),
    Stats(numderivs=1, cache=1),
    UmapDR(components=10, cache=1),
    HDBSCluster(minclustersize=5)
)

if __name__ == "__main__":
    process.run()
