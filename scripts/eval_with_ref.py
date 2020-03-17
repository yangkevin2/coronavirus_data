import numpy as np
import pandas as pd

from argparse import ArgumentParser

def eval_with_ref(pred_path, ref_path, save_path):
    """Evaluate the predicted compounds <pred_path> by finding which are included in the reference <ref_path>.
    
    Args:
      pred_path: path to the the prediction .csv file.
      ref_path: path to the reference .csv file.
      save_path: path to save the output .csv file.
      
    Returns:
      rows in the prediction csv that overlap with the reference.
      
    """
    
    preds = pd.read_csv(pred_path)
    refs = pd.read_csv(ref_path)
    effective = list(
        map((lambda x: x in list(refs['smiles'])), preds['smiles']))
    output = preds[effective]
    output.to_csv(save_path, index=False)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True,
                        help='path to the the prediction .csv file.')
    parser.add_argument('--ref_path', type=str, required=True,
                        help='path to the reference .csv file.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='path to save the output .csv file.')
    args = parser.parse_args()

    eval_with_ref(**vars(args))
