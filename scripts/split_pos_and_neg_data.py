from argparse import ArgumentParser
import csv
from typing import List


def write(smiles: List[str], path: str):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles'])
        for smile in smiles:
            writer.writerow([smile])


def split_pos_and_neg_data(data_path: str, pos_path: str, neg_path: str):
    with open(data_path) as f:
        pos, neg = [], []

        for row in csv.DictReader(f):
            smiles = row['smiles']
            if row['activity'] == '1':
                pos.append(smiles)
            elif row['activity'] == '0':
                neg.append(smiles)
            else:
                raise ValueError(f'Invalid activity "{row["activity"]}"')

    print(f'Num positive = {len(pos):,}')
    write(pos, pos_path)
    print(f'Num negative = {len(neg):,}')
    write(neg, neg_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/binarized_sars.csv',
                        help='Path to binary data .csv file')
    parser.add_argument('--pos_path', type=str, default='../data/sars_pos.csv',
                        help='Path to .csv file where positive data will be written')
    parser.add_argument('--neg_path', type=str, default='../data/sars_neg.csv',
                        help='Path to .csv file where negative data will be written')
    args = parser.parse_args()

    split_pos_and_neg_data(**vars(args))
