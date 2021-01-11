# Description
"""
Segment using clustered segmentation approach and create an audio file per segment
"""

import jinja2
from ftis.analyser import (
    FluidNoveltyslice,
    ClusteredSegmentation,
    ExplodeAudio,
    CollapseAudio
)
from ftis.process import FTISProcess as Chain
from pathlib import Path
from ftis.common.conversion import samps2ms


src = "reaper/highgain_source/bounces"
folder = "outputs/segments"

process = Chain(
    source=src, 
    folder=folder
)

cs = ClusteredSegmentation(numclusters=3, windowsize=5)

process.add(
    FluidNoveltyslice(threshold=0.1,minslicelength=4096, feature=0),
    cs,
    ExplodeAudio()
)

if __name__ == "__main__":
    process.run()

    tracks = {}
    for audio_src, slices in zip(cs.output.keys(), cs.output.values()):
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
