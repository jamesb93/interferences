from ftis.analyser.meta import ClusteredNMF
from ftis.world import World
from ftis.corpus import Corpus

src = "../outputs/classification/4_Split/1"
out = "../dump/layers_extraction"

corpus = Corpus(src)
process = World(sink=out)
corpus >> ClusteredNMF(iterations=200, components=10, cluster_selection_method='leaf')

if __name__ == "__main__":
    process.build(corpus)
    process.run()
