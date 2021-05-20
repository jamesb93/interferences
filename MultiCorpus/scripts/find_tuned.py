from ftis.analyser.descriptor import FluidMFCC, LibroCQT
from ftis.analyser.clustering import KDTree
from ftis.analyser.scaling import Standardise
from ftis.analyser.dr import UMAP
from ftis.analyser.stats import Stats
from ftis.process import FTISProcess as Chain
from ftis.corpus import Corpus
from ftis.common.io import get_duration, write_json
from pathlib import Path
import jinja2

tuned = Corpus("/Users/james/Cloud/Projects/ElectroMagnetic/reaper/Convolutions/anchors/media/07-glued.wav")
db = (
    Corpus("~/Cloud/Projects/DataBending/DataAudioUnique")
    .duration(min_duration=2, max_duration=20)
)

unstatic = Corpus("~/Cloud/Projects/ElectroMagnetic/outputs/classification/4_Split/0")
static = Corpus("~/Cloud/Projects/ElectroMagnetic/outputs/classification/4_Split/1")

output = "../../reaper/Convolutions/tuned"

analysis = Chain(
    source = (db+tuned+unstatic+static),
    folder = output
)

kdtree = KDTree()
dr = UMAP(components=10, cache=1) # we need access to the original data
analysis.add(
    # FluidMFCC(discard=True, numcoeffs=20, fftsettings=[4096, -1, -1], cache=1),
    LibroCQT(cache=0),    
    Stats(numderivs=1, flatten=True, cache=1),
    Standardise(cache=1),
    dr,
    kdtree
)

if __name__ == "__main__":
    analysis.run()

    pinpoint = tuned.items[0] # single item
    x = dr.output[pinpoint]
    dist, ind = kdtree.model.query([x], k=200)
    keys = [x for x in dr.output.keys()]
    names = [keys[x] for x in ind[0]]
    d = {"1" : names}
    write_json(analysis.folder / "nearest_files.json", d)