"""Data from https://www.diamond.ac.uk/covid-19/for-scientists/Main-protease-structure-and-XChem/Downloads.html
(specifically https://www.diamond.ac.uk/dam/jcr:b367c699-6c2b-44d4-b745-2a5927ab8dea/Mpro%20full%20XChem%20screen%20-%20experiment%20summary%20-%20ver-2020-03-16.csv)"""

from argparse import ArgumentParser

import pandas as pd


def process(data_path: str, save_path: str):
    data = pd.read_csv(data_path)
    data = data[['CompoundSMILES', 'RefinementOutcome']]
    data = data[data['CompoundSMILES'].notna() & data['RefinementOutcome'].isin(['5 - Deposition ready', '6 - Deposited', '7 - Analysed & Rejected'])].copy()
    data.loc[data['RefinementOutcome'] != '7 - Analysed & Rejected', 'RefinementOutcome'] = '1'
    data.loc[data['RefinementOutcome'] == '7 - Analysed & Rejected', 'RefinementOutcome'] = '0'
    data = data.rename(columns={'CompoundSMILES': 'smiles', 'RefinementOutcome': 'activity'})
    data.to_csv(save_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../raw_data/mpro_xchem_all.csv',
                        help='Path to raw data')
    parser.add_argument('--save_path', type=str, default='../data/mpro_xchem.csv',
                        help='Path where processed data will be saved')
    args = parser.parse_args()

    process(**vars(args))
