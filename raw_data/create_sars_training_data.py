

with open('../data/binarized_sars.csv', 'w') as wf:
    wf.write('smiles,activity\n')
    with open('../raw_data/active_cid2smiles.txt', 'r') as rf:
        for line in rf:
            line = line.strip().split('\t')[1]
            wf.write(line + ',1' '\n')
    with open('../raw_data/inactive_cid2smiles.txt', 'r') as rf:
        for line in rf:
            line = line.strip().split('\t')[1]
            wf.write(line + ',0' '\n')