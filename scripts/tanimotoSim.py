import rdkit.Chem as Chem
from rdkit import DataStructs

def get_fingerprints(path):
    ret = dict()
    with open(path, 'r') as f:
        f.readline()
        for i, line in enumerate(f.readlines()):
            if i%10000 == 0:
                print(f'Processing {i} for fingerprints.')
            line = line.strip().split(',')
            smile = line[0]
            ret[smile] = [Chem.RDKFingerprint(Chem.MolFromSmiles(smile))]
            if len(line) > 1:
                ret[smile].append(int(line[-1]))
    return ret

def find_most_similar(queries, compare, out):
    with open(out, 'w') as f:
        f.write('smile,training_smile,hit,similarity\n')
        for i, q in enumerate(queries):
            q = q.strip()
            if i%500 == 0:
                print('Processing smile', i)
            best_sim, best_smile = -1, None
            for smile in compare:
                sim = DataStructs.FingerprintSimilarity(queries[q][0], compare[smile][0], \
                        metric=DataStructs.TanimotoSimilarity)
                if sim > best_sim:
                    best_sim = sim
                    best_smile = smile
            # write
            f.write(','.join([q, best_smile, str(compare[best_smile][1]), str(best_sim)]))
            f.write('\n')

if __name__ == '__main__':
    training = get_fingerprints('combined_binarized_sars.csv')
    print('Done reading in training data')

    query = get_fingerprints('external_library.csv')
    find_most_similar(query, training, 'external_similar.csv')
    print('Done with external data')

    query = get_fingerprints('broad_repurposing_library.csv')
    find_most_similar(query, training, 'broad_similar.csv')
    print('Done with Broad data')

    query = get_fingerprints('expanded_external_library.csv')
    find_most_similar(query, training, 'expanded_similar.csv')
