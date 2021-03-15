"""
Analyse each segment in the 'gestural' pool and cluster it
"""

from ftis.world import World
from ftis.corpus import Corpus
from ftis.analyser.flucoma import MFCC
from ftis.analyser.stats import Stats
from ftis.analyser.scaling import Standardise, Normalise
from ftis.analyser.dr import UMAP
from ftis.analyser.clustering import HDBSCAN

src = "../dump/micro_segmentation/3.0-ClusteredSegmentation.ExplodeAudio"
out = "../dump/micro_clustering"
corpus = Corpus(src)
world = World(out)

mfcc = MFCC(cache=True, fftsettings=[1024, 512, 1024])
stats = Stats(numderivs=1, flatten=True)
standardise = Standardise(cache=True)
normalise = Normalise(cache=True)
umap = UMAP(components=6)
hdbscan = HDBSCAN(minclustersize=10)

(
    corpus >> mfcc >> stats >>
    standardise >> normalise >>
    umap >> hdbscan
)

if __name__ == "__main__":
    world.build(corpus)
    world.run()
