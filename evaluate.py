import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# with open("models/curriculum learning models/reward_cl8.pkl", "rb") as f:
# with open("models/specialized/reward_test.pkl", "rb") as f:
def plot_rewards(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    episodes = [d["episode"] for d in data]
    # sum_rewards = [d["sum_leader_reward"] for d in data]
    # sum_rewards = [d["sum_leader_reward"] for d in data]
    all_rewards = [d["leader_rewards_per_action"] for d in data]
    std_rewards = np.array([np.std([reward for sublist in rewards for reward in sublist]) for rewards in all_rewards])
    avg_rewards = np.array([d["avg_leader_reward"] / d["n_agents"] for d in data])
    min_rewards = np.array([d["min_leader_reward"] / d["n_agents"] for d in data])
    max_rewards = np.array([d["max_leader_reward"] / d["n_agents"] for d in data])

    plt.figure(figsize=(14,6))
    plt.plot(episodes, avg_rewards, label="Avg Reward", color='blue')
    # plt.plot(episodes, sum_rewards, label="Sum Reward", color='orange')
    # plt.fill_between(episodes, avg_rewards - std_rewards, avg_rewards + std_rewards, color='lightblue', alpha=0.3, label="Min-Max Range")
    plt.fill_between(episodes, min_rewards, max_rewards, color='lightblue', alpha=0.3, label="Min-Max Range")

    # # Annotate n and k
    # for d in data:
    #     plt.text(d["episode"], max_rewards[d["episode"]-data[0]["episode"]]+0.1, f"(n={d['n_agents']}, k={d['n_leaders']})",
    #              rotation=45, fontsize=8, ha='right', va='bottom')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode-wise Reward Statistics with (n, k) labels")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss(file_path):
    checkpoint = torch.load(file_path)

    # Restore logs if available
    if "logs" not in checkpoint:
        print("[INFO] No training logs found in checkpoint.")
        return

    logs = checkpoint["logs"]
    # print(logs)
    # logs = logs.cpu().data
    print(f"[INFO] Training logs loaded).")

    policy_loss = torch.tensor(logs["policy_loss"]).detach().cpu().numpy()
    value_loss  = torch.tensor(logs["value_loss"]).detach().cpu().numpy()
    entropy     = torch.tensor(logs["entropy"]).detach().cpu().numpy()
    kl          = torch.tensor(logs["kl"]).detach().cpu().numpy()

    # print(value_loss)

    # policy_loss = np.array(logs["policy_loss"])
    # value_loss  = np.array(logs["value_loss"])
    # entropy     = np.array(logs["entropy"])
    # kl          = np.array(logs["kl"])
    plt.figure(figsize=(10, 6))
    # plt.ylim(0, 50)

    plt.plot(policy_loss, label="Policy loss")
    plt.plot(value_loss, label="Value loss")

    plt.xlabel("Training update", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def main():
    plot_rewards("models\\specialized\\reward_test31.pkl")
    plot_rewards("reward_test95.pkl")
    plot_rewards("reward_test98.pkl")
    # plot_rewards("reward_test99.pkl")

    # plot_loss("models\\specialized\\test31.pt")
    # plot_loss("test95.pt")
    # plot_loss("test98.pt")
    # plot_loss("test99.pt")

if __name__ == "__main__":
    main()