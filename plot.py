import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import sys

def load_logs(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    if "logs" not in checkpoint:
        print("[ERROR] No 'logs' key found in checkpoint.")
        sys.exit(1)

    logs = checkpoint["logs"]
    del checkpoint
    print(f"[INFO] Logs loaded from {checkpoint_path}")
    print(f"  - {len(logs.get('episode_rewards', []))} episodes recorded.")
    print(f"  - {len(logs.get('policy_loss', []))} policy updates logged.\n")
    return logs

# ======= Configuration =======
CHECKPOINT_PATH = "ppo_agent_7.pt"   # <-- change to your saved model path
SAVE_FIGURES = False                     # set True if you want to save plots as .png
# ==============================


logs = load_logs(CHECKPOINT_PATH)
# key = "policy_loss"
# title = "Policy Loss"

# key = "value_loss"
# title = "Value Loss"

# key = "entropy"
# title = "Entropy"

key = "kl"
title = "KL Divergence"

# ylabel = "Loss"
# ylabel = "Entropy"
ylabel = "KL Divergence"

save_fig = None

values = logs[key]

if not values:
    print(f"[WARN] No data found for {title}. Skipping...")
else:     
    plt.figure(figsize=(6,4))
    plt.plot(values, linewidth=1.8)
    plt.title(title)
    plt.xlabel("Update Step")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    if save_fig:
        fname = f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=150)
        print(f"[INFO] Saved: {fname}")
    else:
        plt.show()