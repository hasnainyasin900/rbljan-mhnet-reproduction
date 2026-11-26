# RBLJAN & MH-Net Reproduction — Network Traffic Classification

Reproduction of two encrypted-traffic classification models on the USTC-TFC and
ISCX-VPN datasets, by **Hasnain Yasin**. The work was done for a Master's
supervision discussion with Prof. Xiao Xi, Tsinghua Shenzhen.

## Papers

- **RBLJAN** — "RBLJAN: Robust Byte-Label Joint Attention Network for Network
  Traffic Classification", Xiao et al., IEEE Transactions on Dependable and
  Secure Computing, 2025. DOI: 10.1109/TDSC.2024.3478838.
  Official code: https://github.com/ws0407/RBLJAN
- **MH-Net** — "Revolutionizing Encrypted Traffic Classification with MH-Net:
  A Multi-View Heterogeneous Graph Model", Zhang et al., AAAI 2025.
  arXiv:2501.03279. Official code: https://github.com/ViktorAxelsen/MH-Net

Prof. Xiao is a co-author of both papers.

## What is in this repository

- `rbljan/` — my own scripts and Kaggle notebooks for the RBLJAN reproduction
  (preprocessing driver, training wrapper, Modal runner, notebook generator).
  The official RBLJAN code is cloned at run time, not vendored here.
- `mh-net/` — the MH-Net code as run, with my fixes (`config_fixed.py` and the
  `pcap2npy.py` payload fix). Its original README and LICENSE are included.
- `results/` — headline results and full per-class tables (CSV).
- `rbljan/results/` — raw `results.json` and `train.log` from the runs.

Datasets (USTC-TFC, ISCX-VPN) are **not** included — download them from their
official sources. The preprocessed ISCX-VPN packets used here are published
as a Kaggle dataset (link below). The Kaggle notebooks are public:

- Preprocessing (CPU): https://www.kaggle.com/code/hasnainyasin900/rbljan-iscx-vpn-preprocessing-cpu
- GPU training (50 epochs): https://www.kaggle.com/code/hasnainyasin900/rbljan-iscx-vpn-gpu-train-50-epochs
- Preprocessed dataset: https://www.kaggle.com/datasets/hasnainyasin900/rbljan-iscx-vpn-pkl

## Verified results

Single runs (not the papers' 10-fold cross-validation). Full per-class tables
are in `results/`.

| Model | Dataset | Level | Test size | Accuracy | Macro F1 |
|---|---|---|---|---|---|
| RBLJAN | ISCX-VPN | Packet | 21,248 packets | 0.9688 | 0.9700 |
| RBLJAN | USTC-TFC | Packet | 19,968 packets | 0.9608 | 0.9610 |
| MH-Net | USTC-TFC | Flow | 290 flows | 0.9828 | 0.9777 |
| MH-Net | USTC-TFC | Packet | 1,403 packet-graphs | 0.9380 | 0.9333 |
| MH-Net | ISCX-VPN | Flow | 172 flows | 0.9128 | 0.9163 |
| MH-Net | ISCX-VPN | Packet | 1,964 packet-graphs | 0.8931 | 0.8943 |

Run settings: RBLJAN on Kaggle (Tesla T4) — ISCX-VPN 50 epochs, best epoch 45;
USTC-TFC 30 epochs, best epoch 30. MH-Net on Modal (A10G) — 20 epochs,
seed 32, batch 24, lr 1e-2.

## Findings, stated honestly

- **Original test set.** The USTC-TFC test set used here is the paper's
  official split: 20,000 packets, 2,000 per class, 50–1500 bytes, payload-less
  packets kept, single 90/10 split, seed 41.
- **Ceiling.** On that original test set, accuracy is limited to about
  **0.9621**: after masking, many packets are byte-identical but carry
  different labels, so no model can separate them. The reproduction sits at
  that limit (0.9608 in the 30-epoch run).
- **Payload-only variant (not the original set).** Restricting to packets with
  payload longer than 50 bytes (11,772 packets) gives accuracy 0.9890 and
  macro F1 0.9865. This is a *variant* on a different, easier test set — it
  must never be quoted as the original-set result. Its raw output is in
  `rbljan/results/ustc-tfc-payload-only-variant-results.json`.
- **Pipeline bug fixed.** The official `pcap2npy` read payload only from
  scapy's Raw layer. DNS/TLS/HTTP/SMTP packets have no Raw layer, so their
  payloads read as empty and their flows were dropped (Tinba lost all flows).
  The fix reads the transport-layer payload; surviving flows went from 1,248
  to 2,838, and Tinba flow F1 reached 1.0000 after the fix.
- **Gap to the paper.** The RBLJAN paper reports an average F1 of 0.9786 on
  USTC-TFC (10-fold cross-validation, 120 epochs, with GAN augmentation).
  This reproduction is a single run, 30 epochs, without GAN — that is the
  main reason for the gap, together with the ceiling above.

## How to run

RBLJAN (Kaggle): open the preprocessing notebook, then the GPU training
notebook; both attach the preprocessed dataset linked above. Or run locally
with `rbljan/run_rbljan.py` (see the scripts for the exact commands).
MH-Net: see `mh-net/README.md` (original instructions) and use
`config_fixed.py` for the fixed configuration.

## Contact

Hasnain Yasin — https://www.linkedin.com/in/hasnainyasin/
