# combine broad with robert malone's reference positives, and remove those that are in the combined AID training data already. 
# the reference positives are labeled as 1 while the broad library is labeled as 0. 
# We want to check that the model is assigning higher scores to known positives it hasn't seen. 

cid2smiles = {}

reference = set()
with open('../raw_data/rmalone_reference_actives_cid2smiles.txt', 'r') as rf:
    rf.readline()
    for line in rf:
        cid = line.strip().split('\t')[0]
        cid2smiles[cid] = line.strip().split('\t')[1]
        reference.add(cid)

broad = set()
with open('../raw_data/broad_smiles2cid.txt', 'r') as rf:
    rf.readline()
    for line in rf:
        cid = line.strip().split('\t')[-1] # sometimes there is no cid, but that's fine, this will just be a smiles string
        cid2smiles[cid] = line.strip().split('\t')[0]
        broad.add(cid)

training = set()
with open('../conversions/AID1706_training_conversions.csv', 'r') as rf:
    rf.readline()
    for line in rf:
        cid = line.strip().split(',')[0]
        cid2smiles[cid] = line.strip().split(',')[-1]
        training.add(cid)

positives = reference.difference(training)
negatives = broad.difference(reference).difference(training)

with open('../data/evaluation_set_v2.csv', 'w') as wf:
    wf.write('smiles,label\n')
    for s in positives:
        wf.write(cid2smiles[s] + ',1\n')
    for s in negatives:
        wf.write(cid2smiles[s] + ',0\n')