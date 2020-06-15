import os, subprocess, jinja2, random, umap, hdbscan
from flucoma.utils import get_buffer
from flucoma import fluid
from pathlib import Path
from uuid import uuid4
from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
from scipy.stats import moment
import numpy as np


NMFCOMPONENTS = 15

HDBCLUSTSIZE = 2
HDBSAMPS = 1

media = Path("../reaper/source/media/")
source = media / "twovoice.wav"# "06-xbox controller-200518_1319.wav"# "02-200420_0928.wav"
output = Path("slices").resolve()
activation_pickle = Path("activations.wav")

data, labels = [], []

nmf = fluid.nmf(source, activations=activation_pickle.resolve(), iterations=50, components=NMFCOMPONENTS)
activations = get_buffer(nmf.activations, "numpy")
stats = get_buffer(fluid.stats(nmf.activations, numderivs=1), "numpy")

# clustering
print(f'Clustering data')
clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBCLUSTSIZE, min_samples=HDBSAMPS)
cluster_labels = clusterer.fit_predict(stats)
unique_clusters = list(dict.fromkeys(cluster_labels))

sound = get_buffer(nmf.resynth, "numpy")

# Format everything into a lovely little reaper project

# initiate jinja2 business
env = jinja2.Environment(loader=jinja2.FileSystemLoader(['../RPRTemplates']))
template = env.get_template("LayersSquashTemplate.rprtemplate")

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
	"hdbclustersize" : HDBCLUSTSIZE,
	"hdbsamps" : HDBSAMPS
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
	
# create the reaper session as a frontend
with open(reaper_session, "w") as f:
	f.write(template.render(tracks=tracks, metadata=metadata))






