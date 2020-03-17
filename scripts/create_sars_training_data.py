with open('../data/combined_binarized_sars.csv', 'w') as wf:
    wf.write('smiles,activity\n')
    with open('../raw_data/combined_active_cid2smiles.txt', 'r') as rf:
        pos_cids = set()
        for line in rf:
            cid = line.strip().split('\t')[0]
            assert cid not in pos_cids # no dupes
            pos_cids.add(cid)
            smiles = line.strip().split('\t')[1]
            wf.write(smiles + ',1' '\n')
    with open('../raw_data/combined_inactive_cid2smiles.txt', 'r') as rf:
        for line in rf:
            cid = line.strip().split('\t')[0]
            if cid not in pos_cids: # if different assays conflict, bias toward positive TODO is this fine?
                smiles = line.strip().split('\t')[1]
                wf.write(smiles + ',0' '\n')
