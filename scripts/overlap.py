from argparse import ArgumentParser
import csv
from itertools import combinations
import os
from typing import List, Set

from p_tqdm import p_map
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

remover = SaltRemover()


def standardize_smiles(smiles: str) -> str:
    smiles = smiles.replace('\\', '')
    smiles = smiles.replace('/', '')
    smiles = smiles.replace('@', '')
    mol = Chem.MolFromSmiles(smiles)
    res = remover.StripMol(mol, dontRemoveEverything=True)
    smiles = Chem.MolToSmiles(res)

    return smiles


def get_unique_standardized_smiles(path: str, standardize: bool) -> Set[str]:
    with open(path) as f:
        smiles = [row['smiles'] for row in csv.DictReader(f)]
        if standardize:
            smiles = p_map(standardize_smiles, smiles)
        smiles = set(smiles)

    return smiles


def intersection(paths: List[str], standardize: bool):
    name_to_smiles = {os.path.basename(path).replace('.csv', ''): get_unique_standardized_smiles(path, standardize) for path in paths}

    for name_1, name_2 in combinations(name_to_smiles.keys(), r=2):
        smiles_1, smiles_2 = name_to_smiles[name_1], name_to_smiles[name_2]
        intersect = smiles_1 & smiles_2

        print(f'{name_1} num molecules = {len(smiles_1):,}')
        print(f'{name_2} num molecules = {len(smiles_2):,}')
        print(f'{name_1} and {name_2} intersection = {len(intersect):,}')
        print(f'{name_1} percent overlap = {100 * len(intersect) / len(smiles_1):.2f}%')
        print(f'{name_2} percent overlap = {100 * len(intersect) / len(smiles_2):.2f}%')
        print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--paths', nargs='+', type=str,
                        help='Paths to .csv data files to compare')
    parser.add_argument('--standardize', action='store_true', default=False,
                        help='Whether to standardize SMILES strings')
    args = parser.parse_args()

    intersection(**vars(args))
