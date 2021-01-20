"""
Segment using clustered segmentation approach and create an audio file per segment
"""

from ftis.analyser.flucoma import Noveltyslice
from ftis.analyser.meta import ClusteredSegmentation
from ftis.analyser.audio import ExplodeAudio, CollapseAudio
from ftis.common.conversion import samps2ms
from ftis.world import World # a ftis 'world'
from ftis.corpus import Corpus # a corpus object
from pathlib import Path

src = Corpus("../reaper/highgain_source/bounces")
world = World(sink = "../dump/segments")

slicer = Noveltyslice(
    threshold=0.1, 
    minslicelength=4096, 
    feature=0, 
    cache=1,
)

cs = ClusteredSegmentation(numclusters=3, windowsize=5, cache=1)

src >> slicer >> cs >> ExplodeAudio()
world.build(src)

if __name__ == "__main__":
    world.run()

    # tracks = {}
    # for audio_src, slices in zip(cs.output.keys(), cs.output.values()):
    #     f = Path(audio_src).resolve()
    #     track_id = str(f.name)
    #     pos = 0
    #     for i, (start, end) in enumerate(zip(slices, slices[1:])):
    #         start /= 44100
    #         end /= 44100

    #         item = {
    #             "file": f,
    #             "length": end - start,
    #             "start": start,
    #             "position": pos
    #         }
    #         pos += end-start

    #         if track_id in tracks:
    #             tracks[track_id].append(item)
    #         else:
    #             tracks[track_id] = [item]

    #     # make the necessary folders
    #     reaper_session = Path(folder) / "2_VisaluseSlices.rpp"

    #     env = jinja2.Environment(loader=jinja2.FileSystemLoader(['./RPRTemplates']))
    #     template = env.get_template("SegmentationTemplate.rprtemplate")

    #     with open(reaper_session, "w") as f:
    #         f.write(template.render(tracks=tracks, metadata={"none" : "none"}))
