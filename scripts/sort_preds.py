from argparse import ArgumentParser
import csv

def sort_preds(data_path, save_path):
    data = []
    with open(data_path, 'r') as f:
        for row in csv.DictReader(f):
            data.append((row['smiles'], row['activity']))
    data = sorted(data, key=lambda x: float(x[1]), reverse=True)
    with open(save_path, 'w') as f:
        f.write('smiles,activity\n')
        for d in data:
            f.write(','.join(d) + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to .csv file with data to sort')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path where sorted data will be saved as .csv')
    args = parser.parse_args()

    sort_preds(**vars(args))