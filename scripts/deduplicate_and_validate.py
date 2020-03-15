from argparse import ArgumentParser
import csv

from rdkit import Chem


def deduplicate_and_validate(data_path: str, save_path: str):
    smiles_set = set()
    rows = []
    count = duplicates = invalid = 0

    with open(data_path) as f:
        for row in csv.DictReader(f):
            smiles = row['smiles']
            count += 1

            # Check if molecule is new
            if smiles in smiles_set:
                duplicates += 1
                continue
            else:
                smiles_set.add(smiles)

            # Check if molecule can be processed by rdkit
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError('Invalid molecule')
            except Exception:
                invalid += 1
                continue

            # Add molecule
            rows.append(row)

    print(f'Total = {count:,}')
    print(f'Num duplicates = {duplicates:,}')
    print(f'Num invalid = {invalid:,}')
    print(f'Remaining = {len(rows):,}')

    with open(save_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to .csv file with data to deduplicate')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path where deduplicated data will be saved as .csv')
    args = parser.parse_args()

    deduplicate_and_validate(**vars(args))
