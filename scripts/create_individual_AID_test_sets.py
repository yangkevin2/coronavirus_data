import pickle
import os

with open('../splits/combined_binarized_sars/scaffold/split_0.pkl', 'rb') as rf:
    split = pickle.load(rf)
    test_indices = split[2]

smiles = []
with open('../data/combined_binarized_sars.csv', 'r') as rf:
    rf.readline()
    for line in rf:
        smiles.append(line.strip().split(',')[0])

assert len(smiles) == sum([len(s) for s in split])

test_smiles = [smiles[i] for i in test_indices]

cid2smiles = {}
with open('../raw_data/full_training_cid2smiles.txt', 'r') as rf:
    rf.readline()
    for line in rf:
        line = line.strip().split('\t')
        cid2smiles[line[0]] = line[1]

for dataset in ['AID_1706', 'AID_485353', 'AID_652038']:
    active_smiles = set()
    with open('../raw_data/' + dataset + '_datatable_active.csv', 'r') as rf:
        for line in rf:
            try: 
                cid = int(line.strip().split(',')[2]) # if it's an int then we're past the initial garbage lines
                active_smiles.add(cid2smiles[line.strip().split(',')[2]])
            except:
                continue
    with open(os.path.join('../data/individual_AID_test_sets', dataset + '.csv'), 'w') as wf:
        wf.write('smiles,activity\n')
        for s in test_smiles:
            if s in active_smiles:
                wf.write(s + ',1\n')
            else:
                wf.write(s + ',0\n')
    