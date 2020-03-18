from argparse import ArgumentParser
import os
from typing import List

from matplotlib import offsetbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.manifold import TSNE
from tqdm import tqdm

from chemprop.features import get_features_generator
from chemprop.utils import makedirs


def compare_datasets_tsne(smiles_paths: List[str],
                          smiles_col_name: str,
                          colors: List[str],
                          plot_molecules: bool,
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
        # Get SMILES
        new_smiles = pd.read_csv(smiles_path)[smiles_col_name]
        new_smiles = list(new_smiles[new_smiles.notna()])  # Exclude empty strings
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
    scale = 10
    fontsize = 5 * scale
    fig = plt.figure(figsize=(6.4 * scale, 4.8 * scale))
    plt.title('t-SNE using Morgan fingerprint with Jaccard similarity', fontsize=2 * fontsize)
    ax = fig.gca()
    handles = []
    legend_kwargs = dict(loc='upper right', fontsize=fontsize)

    for slc, color, label in zip(slices, colors, labels):
        if plot_molecules:
            # Plots molecules
            handles.append(mpatches.Patch(color=color, label=label))

            for smile, (x, y) in zip(smiles[slc], X[slc]):
                img = Draw.MolsToGridImage([Chem.MolFromSmiles(smile)], molsPerRow=1, subImgSize=(200, 200))
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img), (x, y), bboxprops=dict(color=color))
                ax.add_artist(imagebox)
        else:
            # Plots points
            s = 450 if label == 'sars_pos' else 150
            plt.scatter(X[slc, 0], X[slc, 1], s=s, color=color, label=label)

    if plot_molecules:
        legend_kwargs['handles'] = handles

    plt.legend(**legend_kwargs)
    plt.xticks([]), plt.yticks([])
    plt.savefig(save_path)

    # Plot pairs of sars_pos and other dataset
    if 'sars_pos' in labels:
        pos_index = labels.index('sars_pos')
        for index in range(len(labels) - 1):
            plt.clf()
            fontsize = 50
            plt.figure(figsize=(6.4 * 10, 4.8 * 10))
            plt.title('t-SNE using Morgan fingerprint with Jaccard similarity', fontsize=2 * fontsize)

            plt.scatter(X[slices[index], 0], X[slices[index], 1], s=150, color=colors[index], label=labels[index])
            plt.scatter(X[slices[pos_index], 0], X[slices[pos_index], 1], s=450, color=colors[pos_index], label=labels[pos_index])

            plt.xticks([]), plt.yticks([])
            plt.legend(loc='upper right', fontsize=fontsize)
            plt.savefig(save_path.replace('.png', f'_{labels[index]}.png'))


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
    parser.add_argument('--smiles_col_name', type=str, default='smiles',
                        help='Name of column in data containing SMILES')
    parser.add_argument('--colors', nargs='+', type=str,
                        default=[
                            'red',
                            'green',
                            'orange',
                            'purple',
                            'blue'
                        ],
                        help='Colors of the points associated with each dataset')
    parser.add_argument('--plot_molecules', action='store_true', default=False,
                        help='Whether to plot images of molecules instead of points')
    parser.add_argument('--max_num_per_dataset', type=int, default=10000,
                        help='Maximum number of molecules per dataset; larger datasets will be subsampled to this size')
    parser.add_argument('--save_path', type=str, default='../plots/tsne.png',
                        help='Path to a .png file where the t-SNE plot will be saved')
    args = parser.parse_args()

    compare_datasets_tsne(**vars(args))
