from ftis.analyser import (
    FluidMFCC, Stats, Standardise, HDBSCluster
)
from ftis.process import FTISProcess as Chain
from ftis.corpus import Corpus, PathLoader
from ftis.common.io import get_duration
from pathlib import Path
import jinja2

folder = "../outputs/em_detailed_clustering"
corpus = Corpus("../outputs/em_detailed_segmentation/1_ExplodeAudio").loudness(max_loudness=20)
analysis = Chain(
    source = corpus, 
    folder = folder
)

clustering = HDBSCluster(minclustersize=10, cache=1)
analysis.add(
    FluidMFCC(discard=True, numcoeffs=20, fftsettings=[1024, -1, -1], cache=1),
    Stats(numderivs=1, flatten=True, cache=1),
    Standardise(cache=1),
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
                "length": get_duration(audiofile),
                "start": 0.0,
                "position": pos
            }
            pos += dur

            if track_id in tracks: 
                tracks[track_id].append(item)
            else: 
                tracks[track_id] = [item]

    # Now make the folders
    reaper_session = Path(folder) / "session.rpp"

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(['../../RPRTemplates']))
    template = env.get_template("ClusterTemplate.rprtemplate")

    with open(reaper_session, "w") as f:
        f.write(template.render(tracks=tracks))
