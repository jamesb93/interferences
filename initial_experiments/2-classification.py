# Description
"""
Split by activity into two clusters hopefully rendering sections that are static, versus those that are more gestural or dynamic
"""

from ftis.analyser.descriptor import Flux
from ftis.analyser.stats import  Stats
from ftis.analyser.scaling import Normalise
from ftis.analyser.clustering import AgglomerativeClustering
from ftis.world import World
from ftis.corpus import Corpus
from pathlib import Path
from shutil import copyfile

src = Corpus("../dump/segments/3.0-ClusteredSegmentation.ExplodeAudio") 
output = "../dump/classification"

world = World(sink=output)

cluster = AgglomerativeClustering(numclusters=2)

src >> Flux() >> Stats(numderivs=2) >> Normalise() >> cluster
world.build(src)

if __name__ == "__main__":
    world.run()

    # Now implement a quasi one-shot analyser to copy the sound files to individual directories
    # We will use the directories as a significant progress point from which new analysis will be orchestrated
    out = Path(output) / "4_Split"
    out.mkdir(exist_ok=True)

    for c in cluster.output:
        folder = out / c
        folder.mkdir(exist_ok=True)
        for audio_file in cluster.output[c]:
            dest = folder / Path(audio_file).name
            copyfile(audio_file, dest)


        




    


