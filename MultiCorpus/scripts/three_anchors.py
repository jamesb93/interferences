from ftis.analyser.descriptor import FluidMFCC
from ftis.analyser.clustering import KDTree
from ftis.analyser.scaling import Standardise
from ftis.analyser.dr import UMAP
from ftis.analyser.stats import Stats
from ftis.process import FTISProcess as Chain
from ftis.corpus import Corpus
from ftis.common.io import get_duration
from pathlib import Path
import jinja2

"""
Taking three anchor points:

/Users/james/Cloud/Projects/ElectroMagnetic/outputs/classification/4_Split/1/Mouse_01_1.wav
/Users/james/Cloud/Projects/DataBending/DataAudioUnique/pnacl_public_x86_64_pnacl_llc_nexe_4.wav
/Users/james/Cloud/Projects/ElectroMagnetic/outputs/classification/4_Split/1/RME Face Panel_01_1.wav

Let's produce a KDTree and then put the nearest neighbours of each anchor onto a track.
"""

anchors = [
    '/Users/james/Cloud/Projects/ElectroMagnetic/outputs/classification/4_Split/1/Mouse_01_1.wav',
    '/Users/james/Cloud/Projects/DataBending/DataAudioUnique/pnacl_public_x86_64_pnacl_llc_nexe_4.wav',
    '/Users/james/Cloud/Projects/ElectroMagnetic/outputs/classification/4_Split/1/RME Face Panel_01_1.wav'
]

db = (
    Corpus("~/Cloud/Projects/DataBending/DataAudioUnique")
    .duration(min_duration=2, max_duration=20)
)

em = Corpus("~/Cloud/Projects/ElectroMagnetic/outputs/classification/4_Split/1")

output = "../../reaper/Convolutions/anchors"

analysis = Chain(
    source = (em+db),
    folder = output
)

kdtree = KDTree()
dr = UMAP(components=10, cache=1) # we need access to the original data
analysis.add(
    FluidMFCC(discard=True, numcoeffs=20, fftsettings=[4096, -1, -1], cache=1),
    Stats(numderivs=1, flatten=True, cache=1),
    Standardise(cache=1),
    dr,
    kdtree
)

if __name__ == "__main__":
    analysis.run()

    tracks = {}
    for anchor in anchors:
        pos = 0
        point = dr.output[anchor]
        dist, ind = kdtree.model.query([point], k=25)
        keys = [x for x in dr.output.keys()]
        names = [keys[x] for x in ind[0]]
        names.append(anchor)
        for i in names:
            dur = get_duration(i)
            item = {
                "file": i,
                "length": dur,
                "start": 0.0,
                "position": pos,
                "color" : ""
            }
            pos += dur

            if anchor in tracks: 
                tracks[str(anchor)].append(item)
            else: 
                tracks[str(anchor)] = [item]

    # Now make the folders
    reaper_session = Path(output) / "anchors.rpp"

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(['../../RPRTemplates']))
    template = env.get_template("ClusterColorTemplate.rprtemplate")

    with open(reaper_session, "w") as f:
        f.write(template.render(tracks=tracks))
        