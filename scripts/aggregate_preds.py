from argparse import ArgumentParser

import csv
import pandas as pd
import pubchempy as pcp

def get_cid(smile):
    return pcp.get_compounds(smile, 'smiles')

def aggregate_preds(data_path, sim_path, save_path):
    sim = pd.read_csv(sim_path)
    sim.index = sim['smile']
    sim['cid'] = sim['smile'].apply(get_cid)
    sim = sim.drop('smile', axis=1)
    sim = sim[['training_smile','cid','hit','similarity']]  # Reorder cols
    sim.columns = ['nearest_training_nbr', 'nearest_training_cid', \
            'nearest_training_nbr_hit', 'nearest_training_nbr_tanimoto']

    data = pd.read_csv(data_path)
    data['cid'] = sim['smiles'].apply(get_cid)
    data = data[['smiles','cid','activity']]
    data.columns = ['smiles', 'predicted_activity']
    out = data.join(sim, on=['smiles'], how='left')
    out.sort_values(by=['predicted_activity', 'nearest_training_nbr_hit', 'nearest_training_nbr_tanimoto'],ascending=False, axis=0)

    out.to_csv(save_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to .csv file with data to sort')
    parser.add_argument('--sim_path', type=str, required=True,
                        help='Path where similarity data is stored.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path where sorted data will be saved as .csv')
    args = parser.parse_args()

    agg_preds(**vars(args))

