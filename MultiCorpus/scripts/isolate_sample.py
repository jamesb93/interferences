from ftis.analyser.descriptor import FluidMFCC
from ftis.analyser.clustering import KDTree
from ftis.analyser.scaling import Standardise
from ftis.analyser.audio import CollapseAudio
from ftis.analyser.dr import UMAP
from ftis.analyser.stats import Stats
from ftis.process import FTISProcess as Chain
from ftis.corpus import Corpus
from ftis.common.io import write_json
from pathlib import Path


corpus = (
    Corpus("~/Cloud/Projects/DataBending/DataAudioUnique")
    .duration(min_duration=0.1, max_duration=10)
)

point = Corpus("~/Cloud/Projects/ElectroMagnetic/reaper/Interruptions/media/06-Kindle Off-200513_1547-glued-04.wav")

output = "../outputs/isolate_static"

analysis = Chain(
    source = (point+corpus),
    folder = output
)

kdtree = KDTree()
dr = UMAP(components=10, cache=1)
analysis.add(
    CollapseAudio(),
    FluidMFCC(discard=True, numcoeffs=20, fftsettings=[2048, -1, -1], cache=1),
    Stats(numderivs=1, flatten=True, cache=1),
    Standardise(cache=1),
    dr,
    kdtree
)

if __name__ == "__main__":
    analysis.run()
    pinpoint = point.items[0] # single item
    x = dr.output["/Users/james/Cloud/Projects/ElectroMagnetic/MultiCorpus/outputs/isolate_static/0_CollapseAudio/06-Kindle Off-200513_1547-glued-04.wav"]
    dist, ind = kdtree.model.query([x], k=40)
    keys = [x for x in dr.output.keys()]
    names = [keys[x] for x in ind[0]]
    d = {"1" : names}
    write_json(analysis.folder / "nearest_files.json", d)
