from ftis.analyser.descriptor import FluidMFCC
from ftis.analyser.clustering import AgglomerativeClustering
from ftis.analyser.scaling import Standardise
from ftis.analyser.dr import UMAP
from ftis.analyser.stats import Stats
from ftis.process import FTISProcess as Chain
from ftis.corpus import Corpus
from ftis.common.io import write_json, get_duration
from pathlib import Path
import jinja2

"""
Give me a crude 3 cluster output of all of the 'static' files. Later on we'll find other files similar to these.
We can use both bits of information to help compose.
"""

output = "../../reaper/Convolutions/base_materials"
em = Corpus("~/Cloud/Projects/ElectroMagnetic/outputs/classification/4_Split/1")

analysis = Chain(
    source = (em),
    folder = output
)

dr = UMAP(components=10, cache=1)
clustering = AgglomerativeClustering(numclusters=3)
analysis.add(
    FluidMFCC(discard=True, numcoeffs=20, fftsettings=[2048, -1, -1], cache=1),
    Stats(numderivs=1, flatten=True, cache=1),
    Standardise(cache=1),
    dr,
    clustering
)

if __name__ == "__main__":
    analysis.run()

    tracks = {}
    for cluster, items in clustering.output.items():
        track_id = cluster
        pos = 0
        for audiofile in items:
            
            dur = get_duration(audiofile)
            item = {
                "file": audiofile,
                "length": dur,
                "start": 0.0,
                "position": pos
            }
            pos += dur

            if track_id in tracks: tracks[track_id].append(item)
            else: tracks[track_id] = [item]

    # Now make the folders
    reaper_session = Path(output) / "base_materails.rpp"

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(['../../RPRTemplates']))
    template = env.get_template("ClusterTemplate.rprtemplate")

    with open(reaper_session, "w") as f:
        f.write(template.render(tracks=tracks))
