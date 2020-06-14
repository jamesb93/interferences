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
from scipy.io import wavfile
import hdbscan

#NMF
NMFCOMPONENTS = 50
#MFCC
NUMCOEFFS = 13
#UMAP
UMAPCOMPONENTS = 25 # UMAP Components
NEIGHBOURS = 4  # UMAP neighbours
MINDIST = 0.05  # UMAP minimum distance
PLOT = True
#HDBSCAN
HDBCLUSTSIZE = 3
HDBSAMPS = 1

media = Path("reaper/source/media/")
source = media / "twovoice.wav"# "06-xbox controller-200518_1319.wav"# "02-200420_0928.wav"
output = Path("slices").resolve()

resynth = Path("nmfout.wav").resolve()
features = Path("mfcc.wav").resolve()

data, labels = [], []

if not resynth.exists(): # let's not redo long NMF's each time
	nmf = fluid.nmf(source, resynth=resynth, iterations=50, components=NMFCOMPONENTS)
if not features.exists():
	mfcc = fluid.mfcc(resynth, features=features, minfreq=500, maxfreq=15000, numcoeffs=13)
stats = get_buffer(fluid.stats(features, numderivs=1))

# flatten statistics because its channels...channels...channels
flatstats = []
for i in range(NMFCOMPONENTS):
	offset = i * NUMCOEFFS
	temp = []
	for j in range(NUMCOEFFS):
		for x in stats[j+offset]:
			temp.append(x)
	flatstats.append(temp)
	
# standardise data
print('Standardising Data')
standardise = StandardScaler()
data = np.array(flatstats)
data = standardise.fit_transform(data)

print(f'Reducing to {UMAPCOMPONENTS} dimensions')
redux = UMAP(n_components=UMAPCOMPONENTS, n_neighbors=NEIGHBOURS, min_dist=MINDIST, random_state=42)
embedding = redux.fit(data)
reduced = embedding.transform(data)

# clustering
print(f'Clustering data')
clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBCLUSTSIZE, min_samples=HDBSAMPS)
cluster_labels = clusterer.fit_predict(reduced)
unique_clusters = list(dict.fromkeys(cluster_labels))

sound = get_buffer(resynth, "numpy")

# Format everything into a lovely little reaper project

# initiate jinja2 business
env = jinja2.Environment(loader=jinja2.FileSystemLoader(['.']))
template = env.get_template("RPRLayersSquashTemplate.rpp_t")

# create a session skeleton
session_folder = Path(str(uuid4().hex))
session_folder.mkdir()
reaper_session = session_folder / "session.rpp"
session_media = session_folder / "media"
session_media.mkdir() 

# sum components into new wav files under the reaper session
# also form the dictionary while were here
tracks = {} # dict to contain all the tracks
metadata = {
	"nmfcomponents": NMFCOMPONENTS,
	"mfcccoeffs": NUMCOEFFS,
	"umapcomponents": UMAPCOMPONENTS,
	"umapneighbours": NEIGHBOURS,
	"umapmindist": MINDIST,
} #dict to contain metadata

for x in unique_clusters:
	summed = np.zeros_like(sound[0]) #make an empty numpy array of same size
	output = session_media / f"{x}_components.wav"
	for idx, cluster in enumerate(cluster_labels):
		if cluster == x:
			summed += sound[idx]
	wavfile.write(output.resolve(), 44100, summed)

	tracks[str(x)] = [{
		"file": output.resolve(),
		"position": 0,
		"start": 0.0,
		"length": len(sound[0]) / 44100
	}]
	
print(tracks)
# create the reaper session as a frontend
with open(reaper_session, "w") as f:
	f.write(template.render(tracks=tracks, metadata=metadata))






