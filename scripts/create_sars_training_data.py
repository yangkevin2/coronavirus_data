with open('../data/binarized_sars.csv', 'w') as wf:
    wf.write('smiles,activity\n')
    with open('../raw_data/active_cid2smiles.txt', 'r') as rf:
        pos = set()
        for line in rf:
            smiles = line.strip().split('\t')[1]
            if smiles not in pos:
                wf.write(smiles + ',1' '\n')
                pos.add(smiles)
    with open('../raw_data/inactive_cid2smiles.txt', 'r') as rf:
        neg = set()
        for line in rf:
            smiles = line.strip().split('\t')[1]
            if smiles not in neg:
                wf.write(smiles + ',0' '\n')
                neg.add(smiles)
