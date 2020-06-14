import os
import subprocess
import jinja2
import numpy as np
import umap.plot
import random
from umap import UMAP
from utils import write_json
from flucoma.utils import get_buffer
from flucoma import fluid
from pathlib import Path
from uuid import uuid4
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import hdbscan

COMPONENTS = 15  # UMAP Components
NEIGHBOURS = 7  # UMAP neighbours
MINDIST = 0.1  # UMAP minimum distance
CLUSTERS = 2 # number of clusters to classify
CLUSTER_ALGORITHM = "HDBSCAN"
PLOT = True
#HDBSCAN
HDBCLUSTSIZE = 3
HDBSAMPS = 1

media = Path("reaper/source/media/")
source = media / "02-200420_0928.wav"
output = Path("slices").resolve()

# containers for data and labels
data, labels = [], []

print('Slicing')
slices = get_buffer(
	fluid.noveltyslice(
		source,
		threshold = 0.4,
		fftsettings = [2048, -1, -1]
	)
)

# slices = get_buffer(
# 	fluid.transientslice(
# 		source,
# 		order = 80,
# 		blocksize = 256,
# 		padsize = 128,
# 		skew = 0.0,
# 		threshfwd = 2.0,
# 		threshback = 1.1,
# 		windowsize = 64,
# 		clumplength = 25,
# 		minslicelength = 2048
# 	)
# )
slices = [int(x) for x in slices]
print(f"Analysing {len(slices)} slices")

for i, (start, end) in enumerate(zip(slices, slices[1:])):
	length = end-start

	mfcc = fluid.mfcc(source, 
		fftsettings = [2048, -1, -1],
		startframe = start,
		numframes = length
	)

	stats = get_buffer(
		fluid.stats(mfcc,
			numderivs = 1
		), "numpy"
	)

	data.append(stats.flatten())
	labels.append(f"slice.{i}")

# standardise data
print('Standardising Data')
standardise = StandardScaler()
data = np.array(data)
data = standardise.fit_transform(data)

# dimension reduction
print(f'Reducing to {COMPONENTS} dimensions')
redux = UMAP(n_components=COMPONENTS, n_neighbors=NEIGHBOURS, min_dist=MINDIST, random_state=42)

embedding = redux.fit(data)
reduced = embedding.transform(data)

# if COMPONENTS > 2 and PLOT:
# 	p = umap.plot.interactive(embedding, point_size=2)
# 	umap.plot.show(p)


# clustering
print('Clustering Data')
if CLUSTER_ALGORITHM == "AG":
	cluster = AgglomerativeClustering(
		n_clusters=CLUSTERS).fit(reduced)

if CLUSTER_ALGORITHM == "HDBSCAN":
	cluster = hdbscan.HDBSCAN(min_cluster_size=HDBCLUSTSIZE, min_samples=HDBSAMPS).fit(reduced)

clumped = [] # clumped slices

cur = -2
for i, c in enumerate(cluster.labels_):
	prev = cur
	cur = c
	if cur != prev:
		clumped.append(slices[i])


# Create reaper files to look at the results
print('Generating REAPER file')
tracks = {}
pos = 0		
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

	if source.stem in tracks:
		tracks[source.stem].append(item)
	else:
		tracks[source.stem] = [item]

pos = 0
for i, (start, end) in enumerate(zip(clumped, clumped[1:])):
	start = (start / 44100)
	end = (end / 44100)

	item = {
		"file": source.resolve(),
		"length": end - start,
		"start": start,
		"position": pos
	}
	pos += end-start

	if "clumped" in tracks:
		tracks["clumped"].append(item)
	else:
		tracks["clumped"] = [item]

# make the necessary folders
session_id = str(uuid4().hex)[:8]
experiments = Path("experiments")
layers = experiments/ "layers"
if not layers.exists(): layers.mkdir()
session = layers / session_id
if not session.exists(): session.mkdir()
reaper_session = session / "session.rpp"

# create a dictionary of metadata
metadata = {
	"components" : COMPONENTS,
	"mindist" : MINDIST,
	"clusters" : CLUSTERS,
	"neighbours" : NEIGHBOURS
}

# now create the reaper project
env = jinja2.Environment(loader=jinja2.FileSystemLoader(['./RPRTemplates']))
template = env.get_template("SegmentationTemplate.rprtemplate")

with open(reaper_session, "w") as f:
	f.write(template.render(tracks=tracks, metadata=metadata))

# open it up for ease
subprocess.call(["open", reaper_session])
