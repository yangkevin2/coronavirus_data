import csv
from copy import deepcopy


def process_pseudomonas():
    with open('../raw_data/ecoli.csv', encoding='utf-8-sig') as f:
        raw_ecoli = list(csv.DictReader(f))

    with open('../raw_data/pseudomonas_raw.csv', encoding='utf-8-sig') as f:
        raw_pseudomonas = list(csv.DictReader(f))

    name_to_row = {}
    for row in raw_pseudomonas:
        if row['Activity'] == '':
            row['Activity'] = 'Inactive'

        # Assume active takes precedence over inactive if same name
        if row['Name'] not in name_to_row or row['Activity'] == 'Active':
            name_to_row[row['Name']] = row

    pseudomonas = []
    for e_row in raw_ecoli:
        p_row = deepcopy(name_to_row[e_row['Name']])
        p_row['SMILES'] = e_row['SMILES']
        pseudomonas.append(p_row)

    with open('../raw_data/pseudomonas.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=raw_ecoli[0].keys(), extrasaction='ignore')
        writer.writeheader()
        for row in pseudomonas:
            if 'SMILES' in row:
                writer.writerow(row)

    with open('../data/pseudomonas.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['smiles', 'activity'], extrasaction='ignore')
        writer.writeheader()
        for row in pseudomonas:
            if 'SMILES' in row:
                row = deepcopy(row)
                row['smiles'] = row.pop('SMILES')
                row['activity'] = 1 if row.pop('Activity') == 'Active' else 0
                writer.writerow(row)


if __name__ == '__main__':
    process_pseudomonas()
