from ftis.analyser.descriptor import FluidMFCC, LibroCQT
from ftis.analyser.scaling import Standardise
from ftis.analyser.dr import UMAP
from ftis.analyser.clustering import HDBSCAN
from ftis.analyser.stats import Stats
from ftis.common.io import get_duration
from ftis.process import FTISProcess as Chain
from ftis.corpus import Corpus
from pathlib import Path
import jinja2


db_corpus = (
    Corpus("~/Cloud/Projects/DataBending/DataAudioUnique")
    .duration(min_duration=0.1, max_duration=10)
)

em_corpus = (
    Corpus("../outputs/em_detailed_segmentation/1_ExplodeAudio")
    .duration(min_duration=0.1, max_duration=20)
)
output = "../outputs/multicorpus_exploring"
analysis = Chain(
    source = (db_corpus + em_corpus), 
    folder = output
)

clustering = HDBSCAN(minclustersize=10, cache=1)
analysis.add(
    FluidMFCC(discard=True, numcoeffs=20, fftsettings=[2048, -1, -1], cache=1),
    Stats(numderivs=1, flatten=True, cache=1),
    Standardise(cache=1),
    UMAP(components=10, cache=1),
    clustering
)

if __name__ == "__main__":
    analysis.run()

    tracks = {}
    for cluster, items in clustering.output.items():
        track_id = cluster
        pos = 0
        for audiofile in items:
            color = ""
            if "/Users/james/Cloud/Projects/ElectroMagnetic/" in audiofile:
                color = "23488255 B"
            else:
                color = "33318502 B"
            dur = get_duration(audiofile)
            item = {
                "file": audiofile,
                "length": dur,
                "start": 0.0,
                "position": pos,
                "color" : color
            }
            pos += dur

            if track_id in tracks: 
                tracks[track_id].append(item)
            else: 
                tracks[track_id] = [item]

    # Now make the folders
    reaper_session = Path(analysis.folder) / "session.rpp"

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(['../../RPRTemplates']))
    template = env.get_template("ClusterColorTemplate.rprtemplate")

    with open(reaper_session, "w") as f:
        f.write(template.render(tracks=tracks))
