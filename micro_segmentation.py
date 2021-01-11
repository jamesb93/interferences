# Description
"""
A heavy sub-segmentation of the classified samples from the source.
The idea is to return short transients and micro gestures, wittling down to the micro level as much as is perceptually reasonable.
"""

import jinja2, os
from ftis.analyser import (
    ClusteredSegmentation, 
    FluidOnsetslice, 
    ExplodeAudio,
    UMAP)
from ftis.process import FTISProcess as Chain
from ftis.common.conversion import samps2ms
from pathlib import Path

src = "outputs/classification/4_Split/0"
folder = "outputs/micro_segmentation"

process = Chain(
    source=src, 
    folder=folder
)

initial_segmentation = FluidOnsetslice(threshold=0.3, cache=True)
cluster_segmentation = ClusteredSegmentation(cache=True)

process.add(
    initial_segmentation,
    cluster_segmentation,
    ExplodeAudio(),
)

if __name__ == "__main__":
    process.run()
    # Write out the clustered segemnts to a REAPER file
    tracks = {}
    for audio_src, slices in cluster_segmentation.output.items():
        f = Path(audio_src).resolve()
        track_id = str(f.name)
        pos = 0
        for i, (start, end) in enumerate(zip(slices, slices[1:])):
            start /= 44100
            end /= 44100

            item = {
                "file": f,
                "length": end - start,
                "start": start,
                "position": pos
            }
            pos += end-start

            if track_id in tracks:
                tracks[track_id].append(item)
            else:
                tracks[track_id] = [item]

        # make the necessary folders
        reaper_session = Path(folder) / "2_VisaluseSlices.rpp"

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(['./RPRTemplates']))
        template = env.get_template("SegmentationTemplate.rprtemplate")

        with open(reaper_session, "w") as f:
            f.write(template.render(tracks=tracks, metadata={"none" : "none"}))
