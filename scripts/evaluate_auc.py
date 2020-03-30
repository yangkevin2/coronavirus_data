from argparse import ArgumentParser
from typing import List

import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def prc_auc_score(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def evaluate_auc(true_path: str, pred_path: str):
    true, pred = pd.read_csv(true_path), pd.read_csv(pred_path)
    assert (true['smiles'] == pred['smiles']).all()
    roc_auc = roc_auc_score(true['activity'], pred['activity'])
    prc_auc = prc_auc_score(true['activity'], pred['activity'])
    print(f'ROC-AUC = {roc_auc}')
    print(f'PRC-AUC = {prc_auc}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--true_path', type=str, required=True,
                        help='Path to true .csv file')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to predictions .csv file')
    args = parser.parse_args()

    evaluate_auc(**vars(args))
