import os
import pickle

read_path = 'results/scale_batch_size_ablation'

file = 'pointmass.pkl'

with open(os.path.join(read_path, file), 'rb') as f:
    results = pickle.load(f)

print(results.keys())

