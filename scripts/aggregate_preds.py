from argparse import ArgumentParser

import csv
import pandas as pd
import pubchempy as pcp


def aggregate_preds(data_path, sim_path, data_conversions_path, train_conversions_path, save_path):
    sim = pd.read_csv(sim_path)
    sim = sim[['smile', 'training_smile','hit','similarity']]  # Reorder cols
    sim.columns = ['smiles', 'nearest_training_nbr', \
            'nearest_training_nbr_hit', 'nearest_training_nbr_tanimoto']

    data = pd.read_csv(data_path)
    data = data[['smiles','activity']]
    data.columns = ['smiles', 'predicted_activity']
    data_conversions = pd.read_csv(data_conversions_path)
    data = data.merge(data_conversions, left_on=['smiles'], right_on=['smiles'], how='left')
    out = data.join(sim.set_index('smiles'), on=['smiles'], how='left')
    
    out.sort_values(by=['predicted_activity', 'nearest_training_nbr_hit', 'nearest_training_nbr_tanimoto'],ascending=False, axis=0)

    train_conversions = pd.read_csv(train_conversions_path)
    train_conversions = train_conversions[['smiles', 'cid', 'title']]
    train_conversions.columns = ['train_smiles', 'neighbor_cid', 'neighbor_title']
    out = out.merge(train_conversions, left_on=['nearest_training_nbr'], right_on=['train_smiles'], how='left', validate="many_to_one")
    out = out[['smiles', 'predicted_activity', 'title', 'cid', 'nearest_training_nbr',
       'nearest_training_nbr_hit', 'nearest_training_nbr_tanimoto', 'neighbor_cid', 'neighbor_title']]
    out.to_csv(save_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to .csv file with data to sort')
    parser.add_argument('--sim_path', type=str, required=True,
                        help='Path where similarity data is stored.')
    parser.add_argument('--data_conversions_path', type=str, required=True,
                        help='Path for file converting prediction smiles to name and cid')
    parser.add_argument('--train_conversions_path', type=str, required=True,
                        help='Path for file converting training smiles to name and cid')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path where sorted data will be saved as .csv')
    args = parser.parse_args()

    aggregate_preds(**vars(args))

