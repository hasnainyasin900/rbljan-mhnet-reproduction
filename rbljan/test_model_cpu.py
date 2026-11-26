#!/usr/bin/env python
"""Local CPU smoke test for the RBLJAN classifier (torch 2.15, no GPU).
Validates: imports, model build, forward+loss+backward, dataloader.
NOTE: the official code divides by torch.cuda.device_count() at import
(ZeroDivisionError on CPU-only machines) and OneHot keeps per-device
masks only for CUDA devices, so this test monkey-patches device_count
to 1 and populates the CPU mask manually. On Kaggle GPU neither patch
is needed.
"""
import sys
import torch

# --- patches so the CUDA-era code imports/runs on CPU ---
torch.cuda.device_count = lambda: 1
sys.path.insert(0, '/home/hatch/workspace/rbljan')
import utils  # noqa: E402
import model as RBLJAN_model  # noqa: E402

# OneHot keeps a per-device byte mask only for CUDA devices at import;
# give it the CPU mask so forward() works on CPU.
RBLJAN_model.OneHot.byte_mask['cpu'] = RBLJAN_model.OneHot.byte_mask_cpu

print('torch', torch.__version__)
print('NUM_LABELS', utils.NUM_LABELS, 'CLASSES', utils.CLASSES)
print('EPOCHS', utils.EPOCHS, 'BATCH_SIZE', utils.BATCH_SIZE)

# --- build the classifier exactly like train_RBLJAN_no_GAN does ---
p_e_matrix = utils.get_position_encoding_matrix(utils.EMBEDDING_DIM)[:utils.PKT_MAX_LEN + utils.NGRAM_PAYLOAD]
overall_label_idx = torch.arange(0, utils.NUM_LABELS).long()
clf = RBLJAN_model.RBLJAN_Classifier(p_e_matrix, overall_label_idx)
n_params = sum(p.numel() for p in clf.parameters())
print('RBLJAN_Classifier built, params:', n_params)
print(clf)

# --- tiny forward + loss + backward on a random batch ---
# NOTE: OneHot's mask buffer is hardcoded to BATCH_SIZE, so the batch
# must be exactly 256 (the Loader always yields full 256-packet batches).
torch.manual_seed(0)
B = 256
x = torch.randint(0, 257, (B, utils.PKT_MAX_LEN)).float()
y = torch.randint(0, utils.NUM_LABELS, (B,))
clf.train()
out1, out2 = clf(x)
print('out1 shape', tuple(out1.shape), 'out2 shape', tuple(out2.shape))
assert out1.shape == (B, utils.NUM_LABELS), out1.shape
loss_fn = utils.FocalLoss()
loss1 = loss_fn(out1, y.long())
num_gpus = int(out2.size(0) / utils.NUM_LABELS)
loss2 = torch.nn.functional.cross_entropy(
    out2, torch.tensor(overall_label_idx.tolist() * num_gpus).long())
loss = loss1 + utils.LAMBDA * loss2
print('loss1', float(loss1), 'loss2', float(loss2), 'total', float(loss))
loss.backward()
print('backward OK; grad norm of fc weight:',
      float(clf.fc[0].weight.grad.norm()))
# eval-mode forward (returns attentions as 2nd output)
clf.eval()
with torch.no_grad():
    out1e, att = clf(x)
print('eval forward OK, out1 shape', tuple(out1e.shape),
      'att tuple len', len(att))
print('CPU SMOKE TEST PASSED')
