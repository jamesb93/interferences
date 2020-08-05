from pydub import AudioSegment
from ftis.common.io import read_json, write_json
from random import choice, uniform

CLUSTERS = read_json("outputs/micro_clustering/5_HDBSCLUSTER.json")
urn = list(CLUSTERS["37"])
num_choices = 10 # how many individual samples to use

# Generate selection pool
selection_pool = []
for x in range(num_choices):
    r = choice(urn)
    selection_pool.append(r)
    urn.remove(r)

# Setup constraints
max_length = 10 * 1000 # maximum length of the final result
min_length = 2 * 1000 # minimum length of the final result
length = uniform(min_length, max_length) # generate a random length between min/max
repeat_chance = 0.1 # ten percent
max_repeats = 4
repeat = 0 #Should we repeat?
memory = 4 # way of mitigating the loop from occuring within a certain window of samples
mem = []

# STICK THIS IN A FOR LOOP TO DO MANY GENERATIONS
# Create audiosegment containers

for x in range(100):
    container = AudioSegment.empty() # the container to append to
    concat = AudioSegment.empty() # an empty container to write to
# Now we satisfy the constraints for generating
    while len(container) < length:
        # Make the decision here
        if repeat:
            r = prev
            repeat -= 1
        else:
            r = choice(selection_pool)
            if uniform(0, 1) < repeat_chance:
                repeat = choice(range(max_repeats))

        prev = r # remember


        concat = AudioSegment.from_wav(r)
        container += concat

    container.export(f"outputs/concat/concat_{x}.wav", format="wav")
