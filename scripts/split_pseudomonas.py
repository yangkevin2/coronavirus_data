"""Split pseudomonas data into a train and test set using a scaffold split
so that the test set has exactly X positives and about Y% of the negatives."""

from argparse import ArgumentParser
import random
import os
from typing import Dict, Set

import pandas as pd
from rdkit import Chem

from chemprop.data import scaffold_to_smiles
from chemprop.utils import makedirs


def scaffold_split_num_pos(data_path: str,
                           max_scaffold_size_in_test: int,
                           num_pos_in_test: int,
                           percent_neg_in_test: float,
                           save_dir: str):
    # Load data
    data = pd.read_csv(data_path)
    mols = [Chem.MolFromSmiles(smiles) for smiles in data['smiles']]

    # Determine scaffolds
    scaffold_to_indices: Dict[str, Set[int]] = scaffold_to_smiles(mols, use_indices=True)
    scaffold_to_indices = {scaffold: sorted(indices) for scaffold, indices in scaffold_to_indices.items()}

    # Split scaffolds into those will all positive, all negative, or mixed activity
    pos_scaffolds, mix_scaffolds, neg_scaffolds = [], [], []
    for scaffold, indices in scaffold_to_indices.items():
        activities = {data.iloc[index]['activity'] for index in indices}

        if activities == {1}:
            pos_scaffolds.append(scaffold)
        elif activities == {0}:
            neg_scaffolds.append(scaffold)
        elif activities == {0, 1}:
            mix_scaffolds.append(scaffold)
        else:
            raise ValueError(f'Found activities "{activities}" but should only be 0 or 1')

    # Reproducibility
    random.seed(0)
    pos_scaffolds, mix_scaffolds, neg_scaffolds = sorted(pos_scaffolds), sorted(mix_scaffolds), sorted(neg_scaffolds)

    # Get small scaffolds
    small_pos_scaffolds = [scaffold for scaffold in pos_scaffolds if len(scaffold_to_indices[scaffold]) < max_scaffold_size_in_test]
    small_mix_scaffolds = [scaffold for scaffold in mix_scaffolds if len(scaffold_to_indices[scaffold]) < max_scaffold_size_in_test]
    small_neg_scaffolds = [scaffold for scaffold in neg_scaffolds if len(scaffold_to_indices[scaffold]) < max_scaffold_size_in_test]

    # Put all big scaffolds in train
    train_scaffolds = sorted(set.union(set(pos_scaffolds) - set(small_pos_scaffolds),
                                       set(mix_scaffolds) - set(small_mix_scaffolds),
                                       set(neg_scaffolds) - set(small_neg_scaffolds)))
    test_scaffolds = []

    # Mixed scaffolds (half in train, half in test)
    random.shuffle(small_mix_scaffolds)
    half = len(small_mix_scaffolds) // 2
    train_scaffolds += small_mix_scaffolds[:half]
    test_scaffolds += small_mix_scaffolds[half:]

    # Positive scaffolds (put in test until hit num_pos_in_test, rest in train)
    random.shuffle(small_pos_scaffolds)
    test_indices = sum((scaffold_to_indices[scaffold] for scaffold in test_scaffolds), [])
    num_pos = sum(data.iloc[test_indices]['activity'])

    for scaffold in small_pos_scaffolds:
        scaffold_size = len(scaffold_to_indices[scaffold])
        if num_pos < num_pos_in_test and scaffold_size <= (num_pos_in_test - num_pos):
            test_scaffolds.append(scaffold)
            num_pos += scaffold_size
        else:
            train_scaffolds.append(scaffold)

    # Negative scaffolds (put in test until hit percent_neg_in_test, rest in train)
    random.shuffle(small_neg_scaffolds)
    test_indices = sum((scaffold_to_indices[scaffold] for scaffold in test_scaffolds), [])
    num_neg_in_test = int(percent_neg_in_test * sum(data['activity'] == 0))
    num_neg = sum(data.iloc[test_indices]['activity'])

    for scaffold in small_neg_scaffolds:
        scaffold_size = len(scaffold_to_indices[scaffold])
        if num_neg < num_neg_in_test and scaffold_size <= (num_neg_in_test - num_neg):
            test_scaffolds.append(scaffold)
            num_neg += scaffold_size
        else:
            train_scaffolds.append(scaffold)

    # Get indices
    train_indices = sum((scaffold_to_indices[scaffold] for scaffold in train_scaffolds), [])
    test_indices = sum((scaffold_to_indices[scaffold] for scaffold in test_scaffolds), [])

    # Checks
    train_scaffolds_set, test_scaffolds_set = set(train_scaffolds), set(test_scaffolds)
    assert len(train_scaffolds_set & test_scaffolds_set) == 0
    assert set.union(train_scaffolds_set, test_scaffolds_set) == set(scaffold_to_indices.keys())

    train_indices_set, test_indices_set = set(train_indices), set(test_indices)
    assert len(train_indices_set & test_indices_set) == 0
    assert set.union(train_indices_set, test_indices_set) == set(range(len(data)))

    # Split data
    train, test = data.iloc[train_indices], data.iloc[test_indices]

    # Print statistics
    print('train')
    print(train['activity'].value_counts())
    print('test')
    print(test['activity'].value_counts())

    # Save scaffolds
    makedirs(save_dir)
    train.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(save_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/pseudomonas.csv',
                        help='Path to data .csv file')
    parser.add_argument('--max_scaffold_size_in_test', type=int, default=20,
                        help='Maximum size of a scaffold in the test set')
    parser.add_argument('--num_pos_in_test', type=int, default=20,
                        help='Number of positives to put in the test set')
    parser.add_argument('--percent_neg_in_test', type=float, default=0.1,
                        help='Percent of negatives to put in the test set')
    parser.add_argument('--save_dir', type=str, default='../splits/pseudomonas_scaffold',
                        help='Path to directory where train/test files will be saved')
    args = parser.parse_args()

    scaffold_split_num_pos(**vars(args))
