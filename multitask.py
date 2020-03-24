import os, random
import copy
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

from chemprop.parsing import add_train_args, modify_train_args
from chemprop.models import MoleculeModel
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, split_data
from chemprop.nn_utils import get_activation_function, initialize_weights, compute_gnorm
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, makedirs, save_checkpoint
from chemprop.train import evaluate, evaluate_predictions, predict


def prepare_data(args):
    data = get_data(path=args.data_path, args=args)
    source_data = get_data(path=args.source_data_path, args=args)

    # split train, val, test
    train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args)

    args.num_tasks = train_data.num_tasks()
    args.features_size = train_data.features_size()
    args.train_data_size = len(train_data) 

    print('source data:', len(source_data))
    print('target data:', len(data))

    return train_data, val_data, test_data, source_data


def prepare_model(args):
    args.output_size = args.num_tasks
    inv_model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')
    inv_model.create_encoder(args)  # phi(x), shared across source and target domain
    inv_model.create_ffn(args)  # source function
    inv_model.src_ffn = inv_model.ffn
    inv_model.create_ffn(args)  # target function
    initialize_weights(inv_model)
    return inv_model.cuda()


def forward(inv_model, mol_batch, loss_func, is_source):
    smiles_batch, target_batch = mol_batch.smiles(), mol_batch.targets()
    mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]).cuda()
    targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]).cuda()

    phi_x = inv_model.encoder(smiles_batch)
    if is_source:
        inv_preds = inv_model.src_ffn(phi_x)
    else:
        inv_preds = inv_model.ffn(phi_x)

    inv_pred_loss = loss_func(inv_preds, targets) * mask
    return inv_pred_loss.sum() / mask.sum()


def train(inv_model, src_data, tgt_data, loss_func, inv_opt, args):
    inv_model.train()
    src_data.shuffle()

    new_size = len(tgt_data) / args.batch_size * args.src_batch_size
    new_size = int(new_size)

    src_pos_data = [d for d in src_data if d.targets[0] == 1]
    src_neg_data = [d for d in src_data if d.targets[0] == 0]
    print(len(tgt_data))
    print(len(src_pos_data), len(src_neg_data), new_size)
    src_data = MoleculeDataset(src_pos_data + src_neg_data[:new_size])

    src_data.shuffle()
    tgt_data.shuffle()

    src_iter = range(0, len(src_data), args.src_batch_size)
    tgt_iter = range(0, len(tgt_data), args.batch_size)

    for i, j in zip(src_iter, tgt_iter):
        inv_model.zero_grad()
        src_batch = src_data[i:i + args.src_batch_size]
        src_batch = MoleculeDataset(src_batch)
        src_loss = forward(inv_model, src_batch, loss_func, is_source=True)
        
        tgt_batch = tgt_data[j:j + args.batch_size]
        tgt_batch = MoleculeDataset(tgt_batch)
        tgt_loss = forward(inv_model, tgt_batch, loss_func, is_source=False)

        loss = (src_loss + tgt_loss) / 2
        loss.backward()
        inv_opt[0].step()
        inv_opt[1].step()

        lr = inv_opt[1].get_lr()[0]
        ignorm = compute_gnorm(inv_model)
        print(f'lr: {lr:.5f}, loss: {loss:.4f}, gnorm: {ignorm:.4f}')


def run_training(args, save_dir):
    tgt_data, val_data, test_data, src_data = prepare_data(args)
    inv_model = prepare_model(args)

    print('invariant', inv_model)

    optimizer = build_optimizer(inv_model, args)
    scheduler = build_lr_scheduler(optimizer, args)
    inv_opt = (optimizer, scheduler)

    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    best_score = float('inf') if args.minimize_score else -float('inf')
    best_epoch = 0
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        train(inv_model, src_data, tgt_data, loss_func, inv_opt, args)

        val_scores = evaluate(inv_model, val_data, args.num_tasks, metric_func, args.batch_size, args.dataset_type)
        avg_val_score = np.nanmean(val_scores)
        print(f'Validation {args.metric} = {avg_val_score:.4f}')
        if args.minimize_score and avg_val_score < best_score or not args.minimize_score and avg_val_score > best_score:
            best_score, best_epoch = avg_val_score, epoch
            save_checkpoint(os.path.join(save_dir, 'model.pt'), inv_model, args=args)        

    print(f'Loading model checkpoint from epoch {best_epoch}')
    model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda)
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    test_preds = predict(model, test_data, args.batch_size)
    test_scores = evaluate_predictions(test_preds, test_targets, args.num_tasks, metric_func, args.dataset_type)

    avg_test_score = np.nanmean(test_scores)
    print(f'Test {args.metric} = {avg_test_score:.4f}')
    return avg_test_score


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--source_data_path', required=True)
    parser.add_argument('--src_batch_size', type=int, default=100)
    parser.add_argument('--lambda_e', type=float, default=0.1)

    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    all_test_score = np.zeros((args.num_folds,))
    for i in range(args.num_folds):
        fold_dir = os.path.join(args.save_dir, f'fold_{i}')
        makedirs(fold_dir)
        all_test_score[i] = run_training(args, fold_dir)

    mean, std = np.mean(all_test_score), np.std(all_test_score)
    print(f'{args.num_folds} fold average: {mean:.4f} +/- {std:.4f}')

    
