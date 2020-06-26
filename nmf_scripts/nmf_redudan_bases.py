import subprocess, jinja2, random, umap, hdbscan
from flucoma.utils import get_buffer
from flucoma import fluid
from datetime import date
from pathlib import Path
from uuid import uuid4
from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
from scipy.stats import moment
from datetime import datetime
from scipy.signal import savgol_filter
import numpy as np



NMFCOMPONENTS = 10
HDBCLUSTSIZE = 2
HDBSAMPS = 2

# media = Path("../reaper/source/media/")
media = Path("../reaper/highgain_source/bounces/")
source = media / "highgain_source-002.wav"#"twovoice.wav"# "06-xbox controller-200518_1319.wav"# "02-200420_0928.wav"
bases = Path("bases.wav")
resynth = Path("resynth.wav")

data = []

if not bases.exists() or not resynth.exists():
	nmf = fluid.nmf(source, resynth=resynth, bases=bases, iterations=100, components=NMFCOMPONENTS)
bases = get_buffer(bases, "numpy")
bases_smoothed = np.zeros_like(bases)

# lets smooth the bases a bit
for i, x in enumerate(bases):
	bases_smoothed[i] = savgol_filter(x, 11, 2)
	
# clustering
print(f'Clustering data')
clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBCLUSTSIZE, min_samples=HDBSAMPS)
cluster_labels = clusterer.fit_predict(bases)
unique_clusters = list(dict.fromkeys(cluster_labels))

sound = get_buffer(resynth, "numpy")

# make the necessary folders
now = str(datetime.now()).replace(':', "-")
session = Path(now)
session_media = session / "media"
if not session.exists(): session.mkdir()
if not session_media.exists(): session_media.mkdir()
reaper_session = session / "session.rpp"


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
	output =  session_media / f"{x}_components.wav"
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

print('Generating REAPER file')
# now create the reaper project
env = jinja2.Environment(loader=jinja2.FileSystemLoader(['../RPRTemplates']))
template = env.get_template("LayersSquashTemplate.rprtemplate")

with open(reaper_session, "w") as f:
    f.write(template.render(tracks=tracks, metadata=metadata))






