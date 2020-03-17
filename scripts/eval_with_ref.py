import numpy as np
import pandas as pd

from argparse import ArgumentParser

def eval_with_ref(pred_path, ref_path, save_path, unique_smiles=True):
    """Evaluate the predicted compounds <pred_path> by finding which are included in the reference <ref_path>.
    
    Args:
      pred_path:      path to the the prediction .csv file.
      ref_path:       path to the reference .csv file.
      save_path:      path to save the output .csv file.
      unique_smiles:  if true, only keep one row if there are entries with the same SMILES
      
    Returns:
      csv file of rows in the prediction csv that overlap with the reference. saved to <save_path>.
      
    """
    
    preds = pd.read_csv(pred_path)
    refs = pd.read_csv(ref_path)
    effective = list(
        map((lambda x: x in list(refs['smiles'])), preds['smiles']))
    output = preds[effective].copy()

    if unique_smiles:
      output.drop_duplicates('smiles', keep='first', inplace=True)
      
    output.to_csv(save_path, index=False)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True,
                        help='path to the the prediction .csv file.')
    parser.add_argument('--ref_path', type=str, required=True,
                        help='path to the reference .csv file.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='path to save the output .csv file.')
    parser.add_argument('--unique_smiles', action='store_true', default=True,
                        help='if true, only keep one row if there are entries with the same SMILES')
    args = parser.parse_args()

    eval_with_ref(**vars(args))
