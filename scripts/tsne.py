from argparse import ArgumentParser
import csv
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm

from chemprop.features import get_features_generator
from chemprop.utils import makedirs


def get_smiles(path: str) -> List[str]:
    with open(path) as f:
        smiles = [row['smiles'] for row in csv.DictReader(f)]

    return smiles


def compare_datasets_tsne(smiles_paths: List[str],
                          colors: List[str],
                          max_num_per_dataset: int,
                          save_path: str):
    assert len(smiles_paths) <= len(colors)

    # Random seed for random subsampling
    np.random.seed(1)

    # Genenrate labels based on file name
    labels = [os.path.basename(path).replace('.csv', '') for path in smiles_paths]

    # Load the smiles datasets
    print('Loading data')
    smiles, slices = [], []
    for smiles_path, color in zip(smiles_paths, colors):
        new_smiles = get_smiles(smiles_path)
        print(f'{os.path.basename(smiles_path)}: {len(new_smiles):,}')

        # Subsample if dataset is too large
        if len(new_smiles) > max_num_per_dataset:
            print(f'Subsampling to {max_num_per_dataset:,} molecules')
            new_smiles = np.random.choice(new_smiles, size=max_num_per_dataset, replace=False).tolist()

        slices.append(slice(len(smiles), len(smiles) + len(new_smiles)))
        smiles += new_smiles

    # Compute Morgan fingerprints
    print('Computing Morgan fingerprints')
    morgan_generator = get_features_generator('morgan')
    morgans = [morgan_generator(smile) for smile in tqdm(smiles, total=len(smiles))]

    print('Running t-SNE')
    import time
    start = time.time()
    tsne = TSNE(n_components=2, init='pca', random_state=0, metric='jaccard')
    X = tsne.fit_transform(morgans)
    print(f'time = {time.time() - start}')

    print('Plotting t-SNE')
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    makedirs(save_path, isfile=True)

    plt.clf()
    fontsize = 50
    plt.figure(figsize=(6.4 * 10, 4.8 * 10))
    plt.title('t-SNE using Morgan fingerprint with Jaccard similarity', fontsize=2 * fontsize)

    for slc, color, label in zip(slices, colors, labels):
        s = 450 if label == 'sars_pos' else 150
        plt.scatter(X[slc, 0], X[slc, 1], s=s, color=color, label=label)

    plt.xticks([]), plt.yticks([])
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.savefig(save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--smiles_paths', nargs='+', type=str,
                        default=[
                            '../data/sars_neg.csv',
                            '../data/broad_repurposing_library.csv',
                            '../data/external_library.csv',
                            '../data/expanded_external_library.csv',
                            '../data/sars_pos.csv',
                        ],
                        help='Path to .csv files containing smiles strings (with header)')
    parser.add_argument('--colors', nargs='+', type=str,
                        default=[
                            'red',
                            'green',
                            'orange',
                            'purple',
                            'blue'
                        ],
                        help='Colors of the points associated with each dataset')
    parser.add_argument('--max_num_per_dataset', type=int, default=10000,
                        help='Maximum number of molecules per dataset; larger datasets will be subsampled to this size')
    parser.add_argument('--save_path', type=str, default='../plots/tsne.png',
                        help='Path to a .png file where the t-SNE plot will be saved')
    args = parser.parse_args()

    compare_datasets_tsne(**vars(args))
