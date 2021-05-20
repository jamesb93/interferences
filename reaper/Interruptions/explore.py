from ftis.analyser.flucoma import MFCC
from ftis.analyser.dr import UMAP
from ftis.analyser.clustering import AgglomerativeClustering
from ftis.analyser.scaling import Standardise
from ftis.analyser.stats import Stats
from ftis.corpus import Corpus
from ftis.world import World

corpus = Corpus("delicate_organisations")
world = World(sink="ftis")

(
    corpus >> 
    MFCC(fftsettings=[32, 16, 32]) >> 
    Stats(flatten=True) >> 
    Standardise() >>
    UMAP() >>
    AgglomerativeClustering()
)

if __name__ == "__main__":
    world.build(corpus)
    world.run()