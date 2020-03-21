"""Data from https://www.diamond.ac.uk/covid-19/for-scientists/Main-protease-structure-and-XChem/Downloads.html
(specifically https://www.diamond.ac.uk/dam/jcr:b367c699-6c2b-44d4-b745-2a5927ab8dea/Mpro%20full%20XChem%20screen%20-%20experiment%20summary%20-%20ver-2020-03-16.csv)"""

from argparse import ArgumentParser

import pandas as pd


def process(all_path: str, hits_path: str, save_path: str):
    all_data = pd.read_csv(all_path)
    all_data.rename(columns={'CompoundSMILES': 'smiles'}, inplace=True)
    non_hits = all_data[all_data['smiles'].notna() & (all_data['RefinementOutcome'] == '7 - Analysed & Rejected')]
    non_hits = non_hits[['smiles']]
    non_hits['activity'] = non_hits['non-covalent'] = non_hits['covalent'] = non_hits['dimer'] = non_hits['surface'] = non_hits['xtal-contact'] = [0] * len(non_hits)

    hits = pd.read_csv(hits_path)
    hits.rename(columns={'Compound SMILES': 'smiles'}, inplace=True)
    hits['activity'] = [1] * len(hits)
    hits['non-covalent'] = hits['Site'] == 'A - active'
    hits['covalent'] = hits['Site'] == 'B - active - covalent'
    hits['dimer'] = hits['Site'].str.contains('C - dimer')
    hits['surface'] = hits['Site'].str.contains('D - surface')
    hits['xtal-contact'] = hits['Site'].str.contains('X - xtal contact')
    hits = hits[['smiles', 'activity', 'non-covalent', 'covalent', 'dimer', 'surface', 'xtal-contact']]

    data = pd.concat([hits, non_hits])
    data.drop_duplicates(subset='smiles', keep='first', inplace=True)
    for key in set(data.keys()) - {'smiles'}:
        print(data[key].value_counts())
    data.to_csv(save_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--all_path', type=str, default='../raw_data/mpro_xchem_all.csv',
                        help='Path to raw data of all molecules (hits and non-hits)')
    parser.add_argument('--hits_path', type=str, default='../raw_data/mpro_xchem_hits.csv',
                        help='Path to raw data of hits')
    parser.add_argument('--save_path', type=str, default='../data/mpro_xchem_multitask.csv',
                        help='Path where processed data will be saved')
    args = parser.parse_args()

    process(**vars(args))
