import numpy as np
from sklearn.neighbors import KDTree
from joblib import load
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from random import choice
from ftis.common.io import read_json
from pathlib import Path
app = FastAPI()

# https://fastapi.tiangolo.com/tutorial/cors/
origins = [
  "http://localhost:8080", "http://127.0.0.1:8080",
  "http://localhost:3000", "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fmt(indices):
    return indices.flatten().tolist()[1:] # discard yourself

training = read_json("4_UMAP.json")
keys = [str(Path(k).name) for k in training.keys()]
vals = [v for v in training.values()]

tree = load("5_KDTree.joblib")

@app.get("/query")
async def query():
    maxskips = 100
    skips = 0

    z = 0
    d = {
        "nodes" : [],
        "links" : [],
    }
    for _ in range(maxskips):
        origin = vals[z:z+1]
        _, ind = tree.query(origin, k=10)
        t = choice(fmt(ind))
        node_id = [x["id"] for x in d["nodes"]]
        
        if keys[z] not in node_id:
            node = {"id" : keys[z], "value" : 1}
            d["nodes"].append(node)

        link = {"source" : keys[z], "target" : keys[t]}
        d["links"].append(link)
        z = t # assign the new node as the next origin
        skips+=1
    return d
