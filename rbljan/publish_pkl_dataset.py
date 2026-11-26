#!/usr/bin/env python
"""After Kaggle Notebook A (preprocessing) completes:
  1. Download its version outputs.
  2. Verify the 12 PKLs + counts.json.
  3. Create the private Kaggle dataset `rbljan-iscx-vpn-pkl` for the
     GPU training notebook to attach.
Usage: python publish_pkl_dataset.py [--skip-download]
Requires: KAGGLE_CONFIG_DIR=~/workspace/user/files
"""
import json
import os
import pickle
import subprocess
import sys

KERNEL = 'hasnainyasin900/rbljan-iscx-vpn-preprocessing-cpu'
DATASET_SLUG = 'rbljan-iscx-vpn-pkl'
OUT_DIR = '/home/hatch/workspace/rbljan-prep/kaggle-outputs/prep'
LABELS = ['audio', 'chat', 'file', 'mail', 'streaming', 'voip',
          'vpn-audio', 'vpn-chat', 'vpn-file', 'vpn-mail',
          'vpn-streaming', 'vpn-voip']


def run(cmd, **kw):
    env = dict(os.environ)
    env['KAGGLE_CONFIG_DIR'] = '/home/hatch/workspace/user/files'
    print('+', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, **kw)
    print(r.stdout[-2000:] if r.stdout else '', flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], flush=True)
        raise SystemExit('command failed: ' + ' '.join(cmd))
    return r


def main():
    skip_download = '--skip-download' in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    if not skip_download:
        run(['/home/hatch/workspace/kaggle-cli-venv/bin/kaggle',
             'kernels', 'output', KERNEL, '-p', OUT_DIR])
    # locate the pkl dir (outputs may be nested under rbljan-pkl/)
    pkl_dir = os.path.join(OUT_DIR, 'rbljan-pkl')
    if not os.path.isdir(pkl_dir):
        pkl_dir = OUT_DIR
    print('pkl dir:', pkl_dir)
    print('files:', sorted(os.listdir(pkl_dir)))
    counts = json.load(open(os.path.join(pkl_dir, 'counts.json')))
    total = counts['total_valid']
    print('total valid packets:', total)
    for label in LABELS:
        p = os.path.join(pkl_dir, label + '.pkl')
        assert os.path.isfile(p), 'missing ' + p
        X = pickle.load(open(p, 'rb'))
        assert X, label + ' empty'
        assert all(51 <= len(pkt) <= 1500 for pkt in X[:100])
        print(f'  {label}: {len(X)} packets')
    assert total > 100000, 'suspiciously few packets: %d' % total
    # create the dataset (private) from the pkl dir
    ds_dir = '/home/hatch/workspace/rbljan-prep/kaggle-dataset-pkl'
    os.makedirs(ds_dir, exist_ok=True)
    for f in os.listdir(pkl_dir):
        if f.endswith('.pkl') or f == 'counts.json':
            src = os.path.join(pkl_dir, f)
            dst = os.path.join(ds_dir, f)
            if os.path.abspath(src) != os.path.abspath(dst):
                import shutil
                shutil.copy(src, dst)
    meta = {
        'title': 'RBLJAN ISCX-VPN Preprocessed Packets',
        'id': 'hasnainyasin900/' + DATASET_SLUG,
        'licenses': [{'name': 'CC0-1.0'}],
        'keywords': ['rbljan', 'iscx-vpn', 'traffic-classification'],
    }
    with open(os.path.join(ds_dir, 'dataset-metadata.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    print('dataset dir ready:', ds_dir, sorted(os.listdir(ds_dir)))
    run(['/home/hatch/workspace/kaggle-cli-venv/bin/kaggle',
         'datasets', 'create', '-p', ds_dir])
    print('DATASET PUBLISHED: hasnainyasin900/' + DATASET_SLUG)


if __name__ == '__main__':
    main()
