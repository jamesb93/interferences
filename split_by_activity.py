# Description
"""
Split by activity into two clusters hopefully rendering sections that are static, versus those that are more gestural or dynamic
"""

from ftis.analyser import Flux, Stats, Normalise, AGCluster
from ftis.process import FTISProcess as Chain
from pathlib import Path
from shutil import copyfile

src = "outputs/segments/2_ExplodeAudio"
folder = "outputs/classification"

process = Chain(
    source=src, 
    folder=folder
)

cluster = AGCluster(numclusters=2)

process.add(
    Flux(cache=False),
    Stats(numderivs=2),
    Normalise(),
    cluster
)

if __name__ == "__main__":
    process.run()

    # Now implement a quasi one-shot analyser to copy the sound files to individual directories
    # We will use the directories as a significant progress point from which new analysis will be orchestrated
    out = Path(folder) / "4_Split"
    out.mkdir(exist_ok=True)

    for c in cluster.output:
        folder = out / c
        folder.mkdir(exist_ok=True)
        for audio_file in cluster.output[c]:
            dest = folder / Path(audio_file).name
            copyfile(audio_file, dest)


        




    


