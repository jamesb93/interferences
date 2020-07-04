from ftis.analyser import Flux
from ftis.process import FTISProcess as Chain

src = "reaper/highgain_source/bounces"
folder = "flux"

process = Chain(
    source=src, 
    folder=folder
)

process.add(
    Flux()
)

process.run()
