from ftis.analyser import ClusteredNMF
from ftis.process import FTISProcess as Chain

src = "all_audio"
folder = "extractions"

process = Chain(
    source=src, 
    folder=folder
)

process.add(
    ClusteredNMF(iterations=200,
    components=10,
    cluster_selection_method='leaf')
)

process.run()
