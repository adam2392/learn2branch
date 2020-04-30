import pickle
import json
from pathlib import Path
import gzip

datadir = Path('./data/processed/tsp/100/train/')

datafiles = [x for x in datadir.rglob(".pkl")]
print(datafiles)

with gzip.open(Path(datadir / 'sample_1.pkl').as_posix(), 'rb') as fin:
    tsp_data = pickle.load(fin)


datadir = Path('./data/processed/cauctions/100_500/train/')

datafiles = [x for x in datadir.glob("*.pkl")]
print(datafiles)

with gzip.open(Path(datadir / 'sample_1.pkl').as_posix(), 'rb') as fin:
    cauctions_data = pickle.load(fin)
print(cauctions_data.keys())

print(tsp_data['episode'], cauctions_data['episode'])
print(tsp_data['instance'], cauctions_data['instance'])
print(tsp_data['node_number'], cauctions_data['node_number'])
print(tsp_data['node_depth'], cauctions_data['node_depth'])
print(len(tsp_data['data']), len(cauctions_data['data']))

print(tsp_data['data'][0], cauctions_data['data'][0])