from info_dissemination import InfoDiss
from rl_agents import PPO
from ctde import CTDE_PPO
from tqdm import tqdm
import numpy as np
import torch, random, pickle

def train_rl(num_episodes=1000, save_path="ppo_agent.pt", agent_path=None):
    random.seed(130323)
    # Define environment and RL agent
    state_dim = 15  # based on your get_observation()
    global_dim = 8
    action_dim = 2  # assign or not-assign
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n = 50
    k = 5

    # agent = PPO(state_dim, action_dim, device=device, training=True)
    agent = CTDE_PPO(state_dim, global_dim, action_dim, k, device=device, training=True)
    if agent_path:
        agent.load(agent_path)

    episode_stats = []

    for episode in tqdm(range(num_episodes)):
        # n_agents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # for n in tqdm(n_agents, leave=False, desc="middle loop - n"):
        #     n_leaders = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        #     for k in tqdm(n_leaders, leave=False, desc="inner loop - k"):
        #         k = int(k*n/10)
        # n = random.choice([50, 60, 70, 80, 90, 100])
        # n = random.choice([10, 20, 30, 40])
        # k_fraction = random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
        # k_fraction = random.choice([0.1, 0.2, 0.3, 0.4])
        # n = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        # k_fraction = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        n = 50
        k_fraction = 0.1
        
        # n = random.choice([50, 60, 70, 80, 90, 100])
        # k_fraction = random.choice([0.1, 0.2, 0.3, 0.4])

        k = int(n * k_fraction)

        # Reset InfoDiss environment for this episode
        sim = InfoDiss(
            k=k,
            n_agents=n,
            map_size=100,
            seed=episode,
            assigning_strategy="rl",
            agent=agent, 
            train=True
        )
        print(f"k: {k}, n: {n}")

        sim.run_simulation() 
        
        # Collect rewards only from leaders
        leader_rewards = [uav.episode_reward for uav in sim.uavs if uav.is_leader]
        leader_rewards_per_action = [uav.reward_history for uav in sim.uavs if uav.is_leader]


        episode_stats.append({
            "episode": episode,
            "n_agents": n,
            "n_leaders": k,
            "leader_rewards_per_action": leader_rewards_per_action,
            "avg_leader_reward": np.mean(leader_rewards),
            "max_leader_reward": np.max(leader_rewards),
            "min_leader_reward": np.min(leader_rewards), 
            "sum_leader_reward": np.sum(leader_rewards),
        })

        # Optionally evaluate performance
        # report = sim.evaluate()
        # print(f"[Episode {episode}] Report: {report}")

        # Save every N episodes
        if (episode + 1) % 50 == 0:
            agent.save(save_path)

    # Final save
    agent.save(save_path)

    # Save rewards to pickle
    with open("reward_test104.pkl", "wb") as f:
        pickle.dump(episode_stats, f)

    print(f"[INFO] Leader rewards saved to reward.pkl")

if __name__ == "__main__":
    train_rl(num_episodes=200, save_path="test104.pt")
