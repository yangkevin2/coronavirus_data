from argparse import ArgumentParser
import csv


def deduplicate(data_path: str, save_path: str):
    smiles_set = set()
    rows = []
    count = 0

    with open(data_path) as f:
        for row in csv.DictReader(f):
            smiles = row['smiles']
            count += 1

            if smiles not in smiles_set:
                rows.append(row)
                smiles_set.add(smiles)

    print(f'Total = {count:,}')
    print(f'Unique = {len(smiles_set):,}')
    print(f'Num duplicates = {count - len(smiles_set):,}')

    with open(save_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/external_library.csv',
                        help='Path to .csv file with data to deduplicate')
    parser.add_argument('--save_path', type=str, default='../data/external_library.csv',
                        help='Path where deduplicated data will be saved as .csv')
    args = parser.parse_args()

    deduplicate(**vars(args))
