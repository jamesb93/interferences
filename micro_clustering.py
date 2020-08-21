"""
Analyse each segment in the 'gestural' pool and cluster it
"""

from ftis.analyser import (
    FluidMFCC, 
    HDBSCluster, 
    Stats, 
    UMAP, 
    Standardise,
    Normalise
)
from ftis.process import FTISProcess as Chain

src = "outputs/micro_segmentation/2_ExplodeAudio"
folder = "outputs/micro_clustering"

process = Chain(
    source=src, 
    folder=folder
)

process.add(
    FluidMFCC(cache=True),
    Stats(numderivs=1, flatten=True, cache=False),
    Standardise(cache=False),
    Normalise(cache=False),
    UMAP(components=6),
    HDBSCluster(minclustersize=10)
)

if __name__ == "__main__":
    process.run()
