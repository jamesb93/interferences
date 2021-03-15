from ftis.analyser.flucoma import MFCC
from ftis.analyser.stats import Stats
from ftis.analyser.scaling import Standardise, Normalise
from ftis.analyser.dr import UMAP
from ftis.analyser.clustering import HDBSCAN
from ftis.corpus import Corpus
from ftis.world import World

from pathlib import Path
from ftis.common.io import get_duration

src = "../outputs/micro_segmentation/2_ExplodeAudio"
folder = "../dump/loudest"

corpus = Corpus(src).loudness(min_loudness=75, max_loudness=100)
world = World(sink=folder)

def discard_amp_bin(self):
    self.output = {
        k: v[1:]
        for k, v in self.output.items()
    }

mfcc = MFCC(numcoeffs=20, fftsettings=[8192, 128, 8192])
mfcc.post = discard_amp_bin
stats = Stats(numderivs=1, spec=['median', 'max', 'min', 'stddev', 'mean', 'skewness'], flatten=True)
umap = UMAP(components=2)
clustering = HDBSCAN(minclustersize=10)

(
    corpus >> mfcc >> stats >> umap >> clustering
)


if __name__ == "__main__":
    world.build(corpus)
    world.run()

    # tracks = {}
    # for cluster, items in clustering.output.items():
    #     track_id = cluster
    #     pos = 0
    #     for audiofile in items:
            
    #         dur = get_duration(audiofile)
    #         item = {
    #             "file": audiofile,
    #             "length": get_duration(audiofile),
    #             "start": 0.0,
    #             "position": pos
    #         }
    #         pos += dur

    #         if track_id in tracks: 
    #             tracks[track_id].append(item)
    #         else: 
    #             tracks[track_id] = [item]

    # # Now make the folders
    # reaper_session = Path(folder) / "Quietest.rpp"

    # env = jinja2.Environment(loader=jinja2.FileSystemLoader(['./RPRTemplates']))
    # template = env.get_template("ClusterTemplate.rprtemplate")

    # with open(reaper_session, "w") as f:
    #     f.write(template.render(tracks=tracks))
