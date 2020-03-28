from argparse import ArgumentParser

import pandas as pd
from sklearn.metrics import roc_auc_score


def evaluate_auc(true_path: str, pred_path: str):
    true, pred = pd.read_csv(true_path), pd.read_csv(pred_path)
    assert true['smiles'] == pred['smiles']
    auc = roc_auc_score(true['activity'], pred['activity'])
    print(f'AUC = {auc}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--true_path', type=str, required=True,
                        help='Path to true .csv file')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to predictions .csv file')
    args = parser.parse_args()

    evaluate_auc(**vars(args))
