"""RBLJAN ISCX-VPN training on Modal (FREE-CREDITS FALLBACK ONLY).

Use ONLY if the Kaggle GPU route fails or runs out of quota.
Free tier: $30/month credits; a ~4h A10G run costs ~$4.40, fully inside
the free credits. NOTHING is charged if the workspace spend limit is set
to $30 BEFORE the first run (see USER_STEPS below).

USER_STEPS (the user does these himself; Muse never adds the card):
  1. Sign up at https://modal.com/signup (use his Google account).
  2. In the Modal dashboard, go to Settings -> Billing and set the
     workspace Spend limit to $30. Verify it shows $30 before running.
  3. Add his card on file when Modal asks for a payment method
     (required for GPU use; the $30 spend limit + $30 free credits mean
     $0 is actually charged for this run).
  4. On this machine, run:  modal setup   (paste the token from the
     dashboard) — one-time only.
  5. Upload the preprocessed data (from the Kaggle Notebook A outputs):
         modal volume put rbljan-data /path/to/rbljan-pkl/ /data/
     (creates the volume if missing; ~140 MB).
  6. Run the training (foreground, streams logs):
         modal run modal_rbljan.py --epochs 50
     or the 2-epoch smoke test:
         modal run modal_rbljan.py --smoke
  7. Fetch the outputs when done:
         modal volume get rbljan-outputs / /path/to/rbljan-modal-out/
     (checkpoints/, results.json, train.log).

What this file does (no GPU work happens on import):
  - builds a CUDA image with torch + the RBLJAN deps
  - defines train(), a Modal function on ONE A10G GPU that clones the
    official RBLJAN repo, lays out the PKL data, and runs run_rbljan.py
    (the same wrapper used on Kaggle: checkpoint after EVERY epoch)
  - persists all outputs to the `rbljan-outputs` Volume
"""
import subprocess
import sys

import modal

app = modal.App("rbljan-iscx-vpn")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "scikit-learn",
        "pandas",
        "dpkt",
        "scapy",
    )
    .apt_install("git")
)

data_vol = modal.Volume.from_name("rbljan-data", create_if_missing=True)
out_vol = modal.Volume.from_name("rbljan-outputs", create_if_missing=True)

# run_rbljan.py is the same training wrapper used on Kaggle; it is baked
# into the image so the Modal run needs no local files at switch time.
RUN_SCRIPT = open("/home/hatch/workspace/rbljan-prep/run_rbljan.py").read()


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,  # 6h hard cap; the run should take ~3-4h
    volumes={"/data": data_vol, "/outputs": out_vol},
)
def train(epochs: int = 50, smoke: bool = False):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    work = "/root/RBLJAN"
    subprocess.run(
        ["git", "clone", "-q", "https://github.com/ws0407/RBLJAN", work],
        check=True,
    )
    # lay out the PKL data exactly as utils.get_dataloader expects
    os.makedirs(work + "/data/ISCX-VPN_PKL", exist_ok=True)
    subprocess.run(
        "cp /data/*.pkl " + work + "/data/ISCX-VPN_PKL/ && "
        "ls -la " + work + "/data/ISCX-VPN_PKL/",
        shell=True, check=True,
    )
    # the training wrapper (same file as on Kaggle)
    with open("/root/run_rbljan.py", "w") as fh:
        fh.write(RUN_SCRIPT)
    out_dir = "/outputs/run1"
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "/root/run_rbljan.py", "--out-dir", out_dir]
    if smoke:
        cmd.append("--smoke")
    else:
        cmd += ["--epochs", str(epochs)]
    print("RUNNING:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=work, check=True)
    out_vol.commit()
    print("TRAINING DONE, outputs committed to rbljan-outputs/run1",
          flush=True)
    return {"status": "done", "out_dir": out_dir}


@app.local_entrypoint()
def main(epochs: int = 50, smoke: bool = False):
    print("Submitting RBLJAN training to Modal (A10G)...")
    result = train.remote(epochs=epochs, smoke=smoke)
    print("Result:", result)
