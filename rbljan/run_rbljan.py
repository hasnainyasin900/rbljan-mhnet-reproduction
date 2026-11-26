#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""RBLJAN packet-level training wrapper (scaled reproduction).
Mirrors the official train_RBLJAN_no_GAN() loop exactly (same model,
loss, optimizer, eval) with two additions:
  1. A state_dict checkpoint is saved AFTER EVERY EPOCH
     (./checkpoints/epoch_NNN.pt) — never lose a run to a late crash.
  2. All output is logged to train.log as well as stdout, and a
     results.json is written at the end with the best-epoch metrics.
Usage:
  python run_rbljan.py --epochs 50 [--smoke] [--out-dir .]
  --smoke: 2 epochs only (timing probe; overrides --epochs)
Run from the RBLJAN repo root so ./data/ISCX-VPN_PKL/ and
./tmp_variable/ resolve. Set CUDA_VISIBLE_DEVICES=0 before running to
force a single GPU (avoids DataParallel 'module.' checkpoint prefixes).
"""
import argparse
import datetime
import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

# repo root is the CWD; utils/model are importable from there
from utils import *  # noqa: F401,F403  (NUM_LABELS, EPOCHS, LR, ...)
from model import RBLJAN_Classifier


def log(msg, fh):
    line = '[{}] {}'.format(datetime.datetime.now().strftime('%m:%d-%H:%M:%S'), msg)
    print(line, flush=True)
    fh.write(line + '\n')
    fh.flush()


def save_checkpoint(model, optimizer, epoch, best_acc, best_f1, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc,
        'best_f1': best_f1,
    }, path)


def train(epochs, out_dir, log_fh):
    # --- MH-Net lesson: create every output dir BEFORE training ---
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    model_dir = os.path.join(out_dir, 'model', DATA_MAP[NUM_LABELS])
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log('output dirs ready: {} and {}'.format(ckpt_dir, model_dir), log_fh)

    try:
        with open('./tmp_variable/p_e_matrix_' + str(EMBEDDING_DIM) + '.variable', 'rb') as f1:
            p_e_matrix = pickle.load(f1)[:PKT_MAX_LEN + NGRAM_PAYLOAD]
        log('loaded cached position-encoding matrix', log_fh)
    except FileNotFoundError:
        p_e_matrix = get_position_encoding_matrix(EMBEDDING_DIM)
        log('generated position-encoding matrix', log_fh)

    overall_label_idx = (torch.arange(0, NUM_LABELS)).long()
    if CUDA:
        overall_label_idx = overall_label_idx.cuda(DEVICE)
        p_e_matrix = p_e_matrix.cuda(DEVICE)

    model = RBLJAN_Classifier(p_e_matrix, overall_label_idx)
    log('model built: {} params'.format(
        sum(p.numel() for p in model.parameters())), log_fh)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if CUDA:
        model = model.cuda(DEVICE)
    log('CUDA: {} device {} (count {})'.format(CUDA, DEVICE, DEVICE_COUNT), log_fh)

    train_loader, test_loader = get_dataloader(EXP_PERCENT, TEST_PERCENT, BATCH_SIZE)
    train_loader.load_all_batches()
    gc.collect()
    test_loader.load_all_batches()
    gc.collect()
    num_batch = train_loader.num_batch
    log('data ready: {} train batches, {} test batches (batch {})'.format(
        num_batch, test_loader.num_batch, BATCH_SIZE), log_fh)

    loss_func1 = FocalLoss(alpha=(train_loader.alpha if num_batch > 2000 else None))

    best_acc, best_f1 = 0, 0
    best_epoch, best_cm, best_prf = -1, None, None
    all_f1, all_test_acc = [], []
    s_t = time.time()
    for epoch in range(1, epochs + 1):
        train_total_loss = 0
        ep_t = time.time()
        log('epoch {} start'.format(epoch), log_fh)
        for i in range(num_batch):
            batch_X, batch_y = train_loader.data[i]
            if CUDA:
                batch_X = batch_X.cuda(DEVICE)
                batch_y = batch_y.cuda(DEVICE)
            model.train()
            optimizer.zero_grad()
            out1, out2 = model(batch_X)
            loss1 = loss_func1(out1, batch_y.long())
            num_gpus = int(out2.size(0) / NUM_LABELS)
            # NOTE: the official code calls .cuda() unconditionally here,
            # which crashes on CPU-only machines; guard it (identical
            # behaviour on GPU, where CUDA is True).
            y_label = torch.tensor(overall_label_idx.tolist() * num_gpus).long()
            if CUDA:
                y_label = y_label.cuda(DEVICE)
            loss2 = F.cross_entropy(out2, y_label)
            loss = loss1 + LAMBDA * loss2
            train_total_loss += float(loss.item())
            loss.backward()
            optimizer.step()
            if i % 2000 == 0:
                log('  batch {}/{} ({:.0f}s)'.format(i, num_batch, time.time() - s_t), log_fh)
        log('epoch {} train done in {:.1f}s, avg loss {:.4f}'.format(
            epoch, time.time() - ep_t, train_total_loss / num_batch), log_fh)
        eval_t = time.time()
        test_accuracy, test_c_matrix = evaluate_loss_acc(model, test_loader, True, NUM_LABELS)
        log('epoch {} eval done in {:.1f}s'.format(epoch, time.time() - eval_t), log_fh)
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            log('  test accuracy increased to {:.4f}'.format(best_acc), log_fh)
        all_test_acc.append(test_accuracy)
        P, R, f1 = deal_cm(test_c_matrix)
        all_f1.append(float(__import__('numpy').array(f1).mean()))
        if all_f1[-1] > best_f1:
            best_f1 = all_f1[-1]
            best_epoch = epoch
            best_cm = test_c_matrix.tolist()
            best_prf = {'precision': P, 'recall': R, 'f1': f1}
            log('  test f1 increased to {:.4f}'.format(best_f1), log_fh)
            # best-model save (whole model, like the official save_model)
            best_path = os.path.join(
                model_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') +
                '_best_epoch_{}.pt'.format(epoch))
            with open(best_path, 'wb') as f2:
                torch.save(model, f2)
            log('  best model saved to {}'.format(best_path), log_fh)
        # --- checkpoint after EVERY epoch (never lose a run) ---
        ckpt_path = os.path.join(ckpt_dir, 'epoch_{:03d}.pt'.format(epoch))
        save_checkpoint(model, optimizer, epoch, best_acc, best_f1, ckpt_path)
        log('epoch {} ok: train loss {:.4f}, test acc {:.4f} (best {:.4f}), '
            'test f1 {:.4f} (best {:.4f}), checkpoint {}'.format(
                epoch, train_total_loss / num_batch, test_accuracy, best_acc,
                all_f1[-1], best_f1, ckpt_path), log_fh)
        log('  per-class f1: {}'.format(
            {CLASSES[k]: round(float(f1[k]), 4) for k in range(NUM_LABELS)}), log_fh)
        if epoch > 60 and max(all_f1[-5:]) < best_f1 and max(all_test_acc[-5:]) < best_acc:
            log('early stop at epoch {} (best acc {:.4f}, best f1 {:.4f})'.format(
                epoch, best_acc, best_f1), log_fh)
            break

    total_s = time.time() - s_t
    results = {
        'epochs_run': epoch,
        'epochs_planned': epochs,
        'best_epoch': best_epoch,
        'best_test_accuracy': best_acc,
        'best_test_f1_macro': best_f1,
        'best_per_class': best_prf,
        'best_confusion_matrix': best_cm,
        'all_test_acc': all_test_acc,
        'all_test_f1': all_f1,
        'total_time_s': total_s,
        'classes': CLASSES,
        'num_train_batches': num_batch,
        'num_test_batches': test_loader.num_batch,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'embedding_dim': EMBEDDING_DIM,
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    log('TRAINING DONE in {:.1f}s ({:.1f} min); best epoch {}, acc {:.4f}, '
        'macro F1 {:.4f}'.format(total_s, total_s / 60, best_epoch, best_acc, best_f1),
        log_fh)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--smoke', action='store_true',
                    help='run only 2 epochs (timing probe)')
    ap.add_argument('--out-dir', default='.')
    args = ap.parse_args()
    epochs = 2 if args.smoke else args.epochs
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'train.log')
    with open(log_path, 'w') as log_fh:
        log('run_rbljan.py start: epochs={} smoke={} out_dir={}'.format(
            epochs, args.smoke, out_dir), log_fh)
        log('CUDA_VISIBLE_DEVICES={}'.format(
            os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')), log_fh)
        train(epochs, out_dir, log_fh)
    print('log written to', log_path)


if __name__ == '__main__':
    main()
