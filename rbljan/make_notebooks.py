#!/usr/bin/env python
"""Generate the Kaggle notebooks for the RBLJAN reproduction.
  Notebook A (CPU): rbljan-iscx-vpn-prep — data preprocessing
  Notebook B (GPU): rbljan-iscx-vpn-gpu-train — smoke test / full training
Writes <name>.ipynb + kernel-metadata.json into
~/workspace/rbljan-prep/kaggle/<name>/, ready for `kaggle kernels push`.
"""
import json
import os
import sys

BASE = '/home/hatch/workspace/rbljan-prep'
KAGGLE_DIR = os.path.join(BASE, 'kaggle')
PREP_SCRIPT = open(os.path.join(BASE, 'prep_rbljan.py')).read()
RUN_SCRIPT = open(os.path.join(BASE, 'run_rbljan.py')).read()
DATASET = 'mdmirazulhasan/iscxvpn2016-vpn-nonvpn'


def md_cell(source):
    return {'cell_type': 'markdown', 'metadata': {},
            'source': source.splitlines(keepends=True)}


def code_cell(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': source.splitlines(keepends=True)}


def notebook(cells):
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                           'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10.0'},
        },
        'nbformat': 4,
        'nbformat_minor': 4,
    }


def write_nb(name, nb, metadata):
    d = os.path.join(KAGGLE_DIR, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + '.ipynb'), 'w') as fh:
        json.dump(nb, fh, indent=1)
    with open(os.path.join(d, 'kernel-metadata.json'), 'w') as fh:
        json.dump(metadata, fh, indent=2)
    print('wrote', d)


def metadata(name, title, enable_gpu, dataset_sources, kid=None):
    return {
        'id': kid or ('hasnainyasin900/' + name),
        'title': title,
        'code_file': name + '.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': enable_gpu,
        'enable_internet': True,
        'dataset_sources': dataset_sources,
        'competition_sources': [],
        'kernel_sources': [],
    }


# ================= Notebook A: CPU preprocessing =================
nb_a = notebook([
    md_cell('# RBLJAN — ISCX-VPN preprocessing (CPU, free)\n'
            'Reproduce the official RBLJAN packet preprocessing on the ISCX-VPN '
            '2016 dataset (Kaggle mirror). Output: one .pkl per label '
            '(12 labels) + counts.json. The prep script is byte-identical to '
            'the official data_preprocessing.py output (verified locally on '
            'pcap / pcapng / RAW files) but uses fast dpkt readers + per-file '
            'caps so multi-GB files do not dominate. Label cap: 20,000 '
            'packets/label.'),
    code_cell(
        'import os, sys\n'
        'print(sys.version)\n'
        'SRC = "/kaggle/input/datasets/mdmirazulhasan/iscxvpn2016-vpn-nonvpn"\n'
        'print("input root exists:", os.path.isdir(SRC))\n'
        'for sub in ["NonVPN-PCAP", "VPN-PCAP"]:\n'
        '    d = os.path.join(SRC, sub)\n'
        '    print(sub, len(os.listdir(d)) if os.path.isdir(d) else "MISSING", "files")\n'
        '!df -h /kaggle/working | tail -1\n'
        '!nproc; free -g | head -2'),
    code_cell(
        '# dpkt + scapy are needed for packet parsing (pandas is preinstalled)\n'
        '!pip install -q dpkt scapy 2>&1 | tail -2\n'
        '!python -c "import dpkt, scapy, pandas; print(\'dpkt\', dpkt.__version__, \'| scapy\', scapy.__version__)"'),
    code_cell(
        '# clone the official RBLJAN repo (for the known-ports CSV)\n'
        '!git clone -q https://github.com/ws0407/RBLJAN /kaggle/working/RBLJAN\n'
        '!ls -la /kaggle/working/RBLJAN/data/service-names-port-numbers.csv'),
    code_cell(
        '%%writefile /kaggle/working/prep_rbljan.py\n' + PREP_SCRIPT),
    code_cell(
        '# RUN PREPROCESSING (foreground, CPU-only, no GPU quota used)\n'
        '# label cap 20000 -> ~200-240k packets total across 12 labels\n'
        '!cd /kaggle/working && python prep_rbljan.py '
        '/kaggle/input/datasets/mdmirazulhasan/iscxvpn2016-vpn-nonvpn '
        '/kaggle/working/rbljan-pkl '
        '/kaggle/working/RBLJAN/data/service-names-port-numbers.csv 20000 '
        '2>&1 | tee /kaggle/working/prep.log\n'
        'print("PREP EXIT DONE")'),
    code_cell(
        '# VERIFY outputs: 12 pkls + counts.json, format check\n'
        'import json, os, pickle\n'
        'OUT = "/kaggle/working/rbljan-pkl"\n'
        'print(sorted(os.listdir(OUT)))\n'
        'counts = json.load(open(os.path.join(OUT, "counts.json")))\n'
        'print(json.dumps(counts, indent=1))\n'
        'for label in ["audio","chat","file","mail","streaming","voip",\n'
        '              "vpn-audio","vpn-chat","vpn-file","vpn-mail",\n'
        '              "vpn-streaming","vpn-voip"]:\n'
        '    X = pickle.load(open(os.path.join(OUT, label + ".pkl"), "rb"))\n'
        '    assert X, label + " empty!"\n'
        '    assert all(isinstance(b, int) and 0 <= b <= 255 for pkt in X[:50] for b in pkt)\n'
        '    assert all(51 <= len(p) <= 1500 for p in X[:50])\n'
        '    print(f"{label}: {len(X)} packets, len range "\n'
        '          f"{min(len(p) for p in X)}-{max(len(p) for p in X)}, "\n'
        '          f"first pkt IP src/dst bytes 12-19: {X[0][12:20]}")\n'
        'print("PREP VERIFICATION PASSED")'),
])
write_nb('rbljan-iscx-vpn-prep', nb_a,
         metadata('rbljan-iscx-vpn-prep',
                  'RBLJAN ISCX-VPN Preprocessing (CPU)',
                  enable_gpu=False, dataset_sources=[DATASET]))

# ================= Notebook B: GPU training =================
# mode = 'smoke' (2 epochs, timing probe) or 'full' (N epochs)
def make_nb_b(mode, epochs=50):
    if mode == 'smoke':
        final_cell = code_cell(
            '# SMOKE TEST: 2 epochs on the full data (timing probe only).\n'
            '# Measures real sec/epoch on this GPU so the full run can be sized.\n'
            '!cd /kaggle/working/RBLJAN && CUDA_VISIBLE_DEVICES=0 python '
            'run_rbljan.py --smoke --out-dir /kaggle/working/out-smoke '
            '2>&1 | tee /kaggle/working/smoke.log\n'
            'print("SMOKE DONE")')
        title = 'RBLJAN ISCX-VPN GPU Train (smoke test)'
    else:
        final_cell = code_cell(
            '# FULL TRAINING: {} epochs, checkpoint after EVERY epoch.\n'
            '!cd /kaggle/working/RBLJAN && CUDA_VISIBLE_DEVICES=0 python '
            'run_rbljan.py --epochs {} --out-dir /kaggle/working/out '
            '2>&1 | tee /kaggle/working/train.log\n'
            'print("FULL TRAINING DONE")'.format(epochs, epochs))
        title = 'RBLJAN ISCX-VPN GPU Train ({} epochs)'.format(epochs)
    return notebook([
        md_cell('# RBLJAN — ISCX-VPN packet-level training (GPU)\n'
                'Scaled reproduction: 1 dataset (ISCX-VPN), packet-level only, '
                '1 run (not the paper\'s 10-fold CV), RBLJAN classifier '
                'without GAN augmentation (train_RBLJAN_no_GAN equivalent). '
                'Checkpoints saved after EVERY epoch.'),
        code_cell(
            '# force a single GPU for clean checkpointing (no DataParallel)\n'
            'import os\n'
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
            'import sys, torch\n'
            'print(sys.version)\n'
            'print("torch", torch.__version__, "cuda", torch.cuda.is_available())\n'
            'print("device count", torch.cuda.device_count())\n'
            'if torch.cuda.is_available():\n'
            '    print("GPU:", torch.cuda.get_device_name(0))\n'
            '!nvidia-smi --query-gpu=name,memory.total --format=csv\n'
            '!df -h /kaggle/working | tail -1'),
        code_cell(
            '# inputs: the preprocessed PKL dataset (from Notebook A)\n'
            'import os\n'
            'PKL = "/kaggle/input/datasets/hasnainyasin900/rbljan-iscx-vpn-pkl"\n'
            'print("pkl dir exists:", os.path.isdir(PKL))\n'
            'print(sorted(os.listdir(PKL)) if os.path.isdir(PKL) else "MISSING")\n'
            '!du -sh /kaggle/input/datasets/hasnainyasin900/rbljan-iscx-vpn-pkl 2>/dev/null'),
        code_cell(
            '# clone the official RBLJAN repo (train/model/utils code)\n'
            '!git clone -q https://github.com/ws0407/RBLJAN /kaggle/working/RBLJAN\n'
            '!ls /kaggle/working/RBLJAN/\n'
            '!ls /kaggle/working/RBLJAN/tmp_variable/'),
        code_cell(
            '# lay out the data exactly as utils.get_dataloader expects:\n'
            '# ./data/ISCX-VPN_PKL/<label>.pkl (run from the repo root)\n'
            '!mkdir -p /kaggle/working/RBLJAN/data/ISCX-VPN_PKL\n'
            '!cp /kaggle/input/datasets/hasnainyasin900/rbljan-iscx-vpn-pkl/*.pkl '
            '/kaggle/working/RBLJAN/data/ISCX-VPN_PKL/\n'
            '!ls -la /kaggle/working/RBLJAN/data/ISCX-VPN_PKL/\n'
            '!cat /kaggle/input/datasets/hasnainyasin900/rbljan-iscx-vpn-pkl/counts.json'),
        code_cell(
            '%%writefile /kaggle/working/RBLJAN/run_rbljan.py\n' + RUN_SCRIPT),
        final_cell,
        code_cell(
            '# VERIFY + SUMMARISE the run outputs\n'
            'import json, os\n'
            'for d in ["/kaggle/working/out-smoke", "/kaggle/working/out"]:\n'
            '    if os.path.isdir(d):\n'
            '        print("==", d)\n'
            '        print(sorted(os.listdir(d)))\n'
            '        ck = os.path.join(d, "checkpoints")\n'
            '        if os.path.isdir(ck):\n'
            '            print("checkpoints:", sorted(os.listdir(ck)))\n'
            '        rj = os.path.join(d, "results.json")\n'
            '        if os.path.isfile(rj):\n'
            '            r = json.load(open(rj))\n'
            '            print(json.dumps({k: r[k] for k in [\n'
            '                "epochs_run", "best_epoch", "best_test_accuracy",\n'
            '                "best_test_f1_macro", "total_time_s"]}, indent=1))\n'
            '            print("per-class F1 (best epoch):", r["best_per_class"]["f1"])\n'
            'found = any(os.path.isfile(os.path.join(d, "results.json"))\n'
            '            for d in ["/kaggle/working/out-smoke", "/kaggle/working/out"])\n'
            'if not found:\n'
            '    raise RuntimeError("TRAINING FAILED: no results.json — see smoke.log/train.log")\n'
            'print("SUMMARY DONE")'),
    ]), title


nb_b_smoke, title_smoke = make_nb_b('smoke')
write_nb('rbljan-iscx-vpn-gpu-train', nb_b_smoke,
         metadata('rbljan-iscx-vpn-gpu-train', title_smoke,
                  enable_gpu=True,
                  dataset_sources=['hasnainyasin900/rbljan-iscx-vpn-pkl'],
                  kid='hasnainyasin900/rbljan-iscx-vpn-gpu-train-smoke-test'))

# Full-training version: run as `python make_notebooks.py full <epochs>`
# to overwrite the GPU notebook with the full run (same kernel id).
if len(sys.argv) > 1 and sys.argv[1] == 'full':
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    nb_b_full, title_full = make_nb_b('full', epochs)
    write_nb('rbljan-iscx-vpn-gpu-train', nb_b_full,
             metadata('rbljan-iscx-vpn-gpu-train', title_full,
                      enable_gpu=True,
                      dataset_sources=['hasnainyasin900/rbljan-iscx-vpn-pkl'],
                      kid='hasnainyasin900/rbljan-iscx-vpn-gpu-train-smoke-test'))
    print('FULL notebook ready ({} epochs). Push with:'.format(epochs))
    print('  cd kaggle/rbljan-iscx-vpn-gpu-train && kaggle kernels push -p .')
else:
    print('NOTE: the full-training version is generated after the smoke test,')
    print('once sec/epoch is measured (epochs sized to fit the GPU quota).')
    print('Generate it with: python make_notebooks.py full <epochs>')
