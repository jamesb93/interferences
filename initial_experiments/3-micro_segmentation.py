# Description
"""
A heavy sub-segmentation of the classified samples from the source.
The idea is to return short transients and micro gestures, wittling down to the micro level as much as is perceptually reasonable.
"""
from ftis.world import World
from ftis.corpus import Corpus
from ftis.analyser.meta import ClusteredSegmentation
from ftis.analyser.flucoma import Onsetslice
from ftis.analyser.audio import ExplodeAudio
from ftis.analyser.dr import UMAP
from ftis.common.conversion import samps2ms
from pathlib import Path
from reathon import nodes as reaper

src = Corpus("../dump/classification/4_Split/1") # the suffix number might change, needs to be on the low activity ones
out = "../dump/micro_segmentation"
world = World(sink=out)


initial_segmentation = Onsetslice(threshold=0.3, cache=True)
cluster_segmentation = ClusteredSegmentation(cache=True)

src >> initial_segmentation >> cluster_segmentation >> ExplodeAudio()


if __name__ == "__main__":
    world.build(src)
    world.run()
    # Write out the clustered segemnts to a REAPER file
    session = reaper.Project()

    # Construct tracks  
    for audio_src, slices in cluster_segmentation.output.items():
        p = Path(audio_src)
        source = reaper.Source(file = p.resolve())
        track = reaper.Track(name = p.stem)
        pos = 0
        for i, (start, end) in enumerate(zip(slices, slices[1:])):
            start /= 44100
            end /= 44100
                
            track.add(
                reaper.Item(
                    source,
                    position = float(pos),
                    length = float(end-start)
                )
            )
            pos += end-start
        session.add(track)

    session.write(Path(out) / "2_VisaluseSlices.rpp")