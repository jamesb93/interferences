from ftis.analyser.descriptor import FluidMFCC
from ftis.analyser.scaling import Standardise
from ftis.analyser.dr import UMAP
from ftis.analyser.clustering import HDBSCAN
from ftis.analyser.stats import Stats
from ftis.process import FTISProcess as Chain
from ftis.corpus import Corpus


db_corpus = (
    Corpus("~/Cloud/Projects/DataBending/DataAudioUnique")
    .duration(min_duration=0.03)
)

em_corpus = (
    Corpus("../outputs/em_detailed_segmentation/1_ExplodeAudio")
)

analysis = Chain(
    source = db_corpus + em_corpus, 
    folder = "../outputs/multicorpus_exploring"
)


analysis.add(
    FluidMFCC(discard=True, numcoeffs=20, fftsettings=[2048, -1, -1], cache=1),
    Stats(numderivs=1, flatten=True, cache=1),
    Standardise(),
    UMAP(components=2),
    HDBSCAN(minclustersize=5)
)

if __name__ == "__main__":
    analysis.run()
