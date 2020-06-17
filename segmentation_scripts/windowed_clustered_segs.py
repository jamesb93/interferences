import numpy as np
import os, subprocess, jinja2, random, hdbscan
from umap import UMAP
from flucoma.utils import get_buffer
from flucoma import fluid
from pathlib import Path
from uuid import uuid4
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from datetime import date

COMPONENTS = 4  # UMAP Components
NEIGHBOURS = 7  # UMAP neighbours
MINDIST = 0.1  # UMAP minimum distance
THRESHOLD = 0.47
WINDOWSIZE = 10
HOPSIZE = 1

media = Path("../reaper/source/media/")
source = media / "02-200420_0928.wav"
source = source.resolve()
output = Path("slices").resolve()

print('Slicing')
slices = get_buffer(
	fluid.noveltyslice(
		source,
		threshold = THRESHOLD,
		fftsettings = [2048, -1, -1]
	)
)

slices = [int(x) for x in slices]

# clustering
standardise = StandardScaler()
original_slices = list(slices) # make a templated copy
tracks = {}

for nclusters in range(2, WINDOWSIZE):
    model = AgglomerativeClustering(n_clusters=nclusters)
    count = 0
    slices = original_slices[:] # recopy the original so we start fresh
    while (count + WINDOWSIZE) <= len(slices):
        indices = slices[count:count+WINDOWSIZE] #create a section of the indices in question
        data = []
        for i, (start, end) in enumerate(zip(indices, indices[1:])):

            mfcc = fluid.mfcc(source, 
                fftsettings = [2048, -1, -1],
                startframe = start,
                numframes = end-start)

            stats = get_buffer(
                fluid.stats(mfcc,
                    numderivs = 1
                ), "numpy")

            data.append(stats.flatten())

        data = standardise.fit_transform(data)

        # might not be necessary to reduce as the dimensions are already quite low
        # redux = UMAP(n_components=COMPONENTS, n_neighbors=NEIGHBOURS, min_dist=MINDIST, random_state=42).fit_transform(data)

        cluster = model.fit(data)
        print(f"num slices {len(slices)}")
        
        cur = -2
        for j, c in enumerate(cluster.labels_):
            prev = cur
            cur = c
            if cur == prev:
                try:
                    slices.pop(j + count)
                except IndexError:
                    print(f"Error at {j}")
                    print(f"Count {count}")
                    print(len(slices))

        count += 1


    # Create reaper files to look at the results
    
    pos = 0
    track_id = str(nclusters)
    for i, (start, end) in enumerate(zip(slices, slices[1:])):
        start = (start / 44100)
        end = (end / 44100)

        item = {
            "file": source.resolve(),
            "length": end - start,
            "start": start,
            "position": pos
        }
        pos += end-start

        if nclusters in tracks:
            tracks[nclusters].append(item)
        else:
            tracks[nclusters] = [item]

# make the necessary folders
today = date.today()
now = today.strftime("%d-%m-%Y")
session_id = str(uuid4().hex)[:5]
session = Path(f"{now}-{session_id}")
if not session.exists(): session.mkdir()
reaper_session = session / "session.rpp"

# create a dictionary of metadata
metadata = {
    "components" : COMPONENTS,
    "mindist" : MINDIST,
    "neighbours" : NEIGHBOURS,
    "threshold" : THRESHOLD,
    "note" : "Windowed clustering",
    "window" : WINDOWSIZE
}

print('Generating REAPER file')
# now create the reaper project
env = jinja2.Environment(loader=jinja2.FileSystemLoader(['../RPRTemplates']))
template = env.get_template("SegmentationTemplate.rprtemplate")

with open(reaper_session, "w") as f:
    f.write(template.render(tracks=tracks, metadata=metadata))
