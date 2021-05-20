from ftis.analyser.descriptor import FluidMFCC
from ftis.analyser.clustering import KDTree
from ftis.analyser.scaling import Standardise
from ftis.analyser.audio import CollapseAudio
from ftis.analyser.dr import UMAP
from ftis.analyser.stats import Stats
from ftis.process import FTISProcess as Chain
from ftis.corpus import Corpus
from ftis.common.io import write_json
from flucoma.dataset import pack
from pathlib import Path

"""
Let's make a KDTree of some convolution candidates to explore in max explorer.maxpat
"""


db = (
    Corpus("~/Cloud/Projects/DataBending/DataAudioUnique")
    .duration(min_duration=0.3, max_duration=20)
)

em = Corpus("~/Cloud/Projects/ElectroMagnetic/outputs/classification/4_Split/1")

output = "../outputs/convolution_candidates"

analysis = Chain(
    source = (em+db),
    folder = output
)

kdtree = KDTree()
dr = UMAP(components=10, cache=1)
analysis.add(
    FluidMFCC(discard=True, numcoeffs=20, fftsettings=[2048, -1, -1], cache=1),
    Stats(numderivs=1, flatten=True, cache=1),
    Standardise(cache=1),
    dr,
    kdtree
)

if __name__ == "__main__":
    analysis.run()
    d = dr.output
    path_out = Path(output) / "dataset.json"
    write_json(path_out, pack(d)) 
