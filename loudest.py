from ftis.analyser import FluidMFCC, Stats, Standardise, UMAP, Normalise, HDBSCluster
from ftis.common.io import get_duration
from ftis.corpus import CorpusLoader, CorpusFilter
from ftis.process import FTISProcess as Chain
from pathlib import Path
import jinja2

src = "outputs/micro_segmentation/2_ExplodeAudio"
folder = "outputs/loudest"

process = Chain(
    source=src, 
    folder=folder
)
clustering = HDBSCluster(minclustersize=10)
process.add(
    CorpusLoader(cache=1),
    CorpusFilter(max_loudness=100, min_loudness=75),
    FluidMFCC(discard=True, numcoeffs=20, fftsettings=[8192, 128, 8192]),
    Stats(numderivs=1, 
        spec = ["median", "max", "min", "stddev", "mean", "skewness"]),
    UMAP(components=2),
    clustering
)

if __name__ == "__main__":
    process.run()

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
    reaper_session = Path(folder) / "Quietest.rpp"

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(['./RPRTemplates']))
    template = env.get_template("ClusterTemplate.rprtemplate")

    with open(reaper_session, "w") as f:
        f.write(template.render(tracks=tracks))
