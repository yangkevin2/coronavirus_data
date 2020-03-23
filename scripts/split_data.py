from argparse import ArgumentParser
import csv
import os
from typing import Tuple

from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.utils import split_data
from chemprop.utils import makedirs
from fire import Fire
from tap import Tap
from tqdm import tqdm


def run_split_data(data_path: str,
                   split_type: str,
                   split_sizes: Tuple[int, int, int],
                   seed: int,
                   save_dir: str):
    with open(data_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        lines = list(reader)

    data = []
    for line in tqdm(lines):
        datapoint = MoleculeDatapoint(line=line)
        datapoint.line = line
        data.append(datapoint)
    data = MoleculeDataset(data)

    train, dev, test = split_data(
        data=data,
        split_type=split_type,
        sizes=split_sizes,
        seed=seed
    )

    makedirs(save_dir)

    for name, dataset in [('train', train), ('dev', dev), ('test', test)]:
        with open(os.path.join(save_dir, f'{name}.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for datapoint in dataset:
                writer.writerow(datapoint.line)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to .csv containing data')
    parser.add_argument('--split_type', type=str, choices=['random', 'scaffold_balanced'], default='random',
                        help='Method of splitting the data')
    parser.add_argument('--split_sizes', type=int, nargs=3, default=(0.8, 0.1, 0.1),
                        help='Size of train/dev/test splits')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory where splits will be saved')
    args = parser.parse_args()

    run_split_data(**vars(args))
