# Description
"""
Analyse each item belonging to a cluster so that they can be ranked in order of a descriptor.
"""

from ftis.analyser import Flux, FluidLoudness, Stats
from ftis.process import FTISProcess as Chain

src = "outputs/micro_segmentation/2_ExplodeAudio"

flux_chain = Chain(
    source=src, 
    folder="outputs/micro_segmentation_flux",
)

loud_chain = Chain(
    source=src,
    folder="outputs/micro_segmentation_loudness"
)

flux = Flux()
loud = FluidLoudness()

flux_chain.add(
    flux,
    Stats(numderivs=0)
)

loud_chain.add(
    loud,
    Stats(numderivs=0)
)

if __name__ == "__main__":
    flux_chain.run()
    loud_chain.run()
