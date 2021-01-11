from ftis.analyser import ClusteredNMF
from ftis.process import FTISProcess as Chain

src = "outputs/classification/4_Split/1"
folder = "outputs/layers_extractions"

process = Chain(
    source=src, 
    folder=folder
)

process.add(
    ClusteredNMF(
        iterations=200,
        components=10,
        cluster_selection_method='leaf'
    )
)

if __name__ == "__main__":
    process.run()
