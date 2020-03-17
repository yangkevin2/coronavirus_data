import pandas as pd 

conv1 = pd.read_csv('../raw_data/full_training_cid2smiles.txt', sep='\t')
conv2 = pd.read_csv('../raw_data/full_training_cid2title.txt', sep='\t')
combined = conv1.merge(conv2)
combined = combined[['smiles', 'title', 'cid']]
combined.to_csv('../conversions/combined_training.csv')