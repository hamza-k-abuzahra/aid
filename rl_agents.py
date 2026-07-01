from logging import logProcesses
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence


ACTION_ASSIGN = 1
ACTION_THRESHOLD = 0.5

class RLAgent:
    def __init__(self):
        pass

    def policy(self, state, determenistic):
        return NotImplementedError

    def observe(self, state, action, reward, done):
        return NotImplementedError
    
    def is_assign(self, action):
        return NotImplementedError
    

#################################################################################################
######################################### IPPO Classes ##########################################
#################################################################################################
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64]):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.shared_layers = nn.Sequential(*layers)

        self.policy_head = nn.Linear(input_dim, action_dim)  # outputs logits for actions
        self.value_head = nn.Linear(input_dim, 1)     
        
        self._init_weights()       # outputs state value

    def _init_weights(self):
        # Initialize last layer to near-zero weights
        nn.init.constant_(self.policy_head.weight, 0.0)
        nn.init.constant_(self.policy_head.bias, 0.0)

        nn.init.constant_(self.value_head.weight, 0.0)
        nn.init.constant_(self.value_head.bias, 0.0)

        
        # For hidden layers, use small initialization
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)  # Small
                nn.init.constant_(layer.bias, 0.0)


    def forward(self, x):
        x = self.shared_layers(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)
 

class RolloutBuffer:
    def __init__(self, buffer_size, state_dim):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.full = False
        # self.state_dim = state_dim

        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32)
        self.actions = torch.zeros(buffer_size, dtype=torch.long)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.next_values = torch.zeros(buffer_size, dtype=torch.float32)

    def add(self, state, action, log_prob, reward, done, value, next_value):
        self.states[self.ptr] = torch.tensor(state, dtype=torch.float32)
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.next_values[self.ptr] = next_value

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True

    def clear(self):
        self.ptr = 0
        self.full = False
        # self.__init__(self.buffer_size, self.state_dim)


class PPO(RLAgent):
    def __init__(self, state_dim, action_dim=2, lr=2e-4, gamma=0.2, clip_epsilon=0.2, 
                 buffer_size=128, update_epochs=5, batch_size=32, device='cpu', training=False):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.training = training

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = RolloutBuffer(buffer_size, state_dim)
        
        self.logs = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
        }

    def train_mode(self):
        self.training = True
        self.model.train()

    def eval_mode(self):
        self.training = False
        self.model.eval()
        
    def policy(self, state, determenistic=False):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits, value = self.model(state_tensor)
        dist = Categorical(logits=logits)
        # print(f"logits: {logits}, value: {value}")
        # print("---- debug PPO policy ----")
        # print("state:", state_tensor)
        # print("dist probs:", dist.probs )
        
        if determenistic: 
            action = torch.argmax(logits)
        else: 
            action = dist.sample()
        

        log_prob = dist.log_prob(action)
        
        # print("action:", action)
        # print("log-prob:", log_prob)
        # print("value:", value)

        return action.detach(), log_prob.detach(), value.detach() 

    def is_assign(self, action):
        action, log_prob, value = action
        return int(action.item()) == 1
    
    def observe(self, state, action, reward, done, next_value):
        if not self.training: return 

        action, log_prob, value = action
        
        self.buffer.add(state, action, log_prob, reward, done, value, next_value)
        # print(f"state: {state}"
        #       f"action: {action}"
        #       f"log_prob: {log_prob}"
        #       f"reward: {reward}"
        #       f"done: {done}"
        #       f"value: {value}")
        if self.buffer.full:
            print("Buffer is full, updating the agent...")
            self.update()

    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        adv = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                # next_value = values[t + 1]
                next_value = next_values[t]
            nextnonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * nextnonterminal - values[t] # use next_values[t]?
            adv[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        returns = adv + values # [:-1]
        return adv, returns

    def update(self):
        self.model.train()
        states = self.buffer.states[:self.buffer.ptr]
        actions = self.buffer.actions[:self.buffer.ptr]
        old_log_probs = self.buffer.log_probs[:self.buffer.ptr]
        rewards = self.buffer.rewards[:self.buffer.ptr]
        dones = self.buffer.dones[:self.buffer.ptr]
        values = self.buffer.values[:self.buffer.ptr]
        next_values = self.buffer.next_values[:self.buffer.ptr]

        assigns = sum(actions)
        print(f"actions: ({assigns}/{self.buffer.ptr})")

        advantages, returns = self.compute_gae(rewards, values, next_values, dones, self.gamma)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses, entropies, kls = [], [], [], []

        print("returns mean:", returns.mean().item())
        print("returns std:", returns.std().item())
        print("values mean:", values.mean().item())
        print("values std:", values.std().item())
        print("advantages std:", advantages.std().item())
        
        for _ in range(self.update_epochs):
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                
                batch_states = states[start:end].to(self.device)
                batch_actions = actions[start:end].to(self.device)
                batch_old_log_probs = old_log_probs[start:end].to(self.device)
                batch_advantages = advantages[start:end].to(self.device)
                batch_returns = returns[start:end].to(self.device)

                logits, values = self.model(batch_states)
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(batch_actions)

                # Approx. KL divergence
                approx_kl = (batch_old_log_probs - new_log_probs).mean()


                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)

                loss = policy_loss + 0.1 * value_loss - 0.02 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                            max_norm=0.5
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                kls.append(approx_kl.item())
                

        self.logs["policy_loss"].append(sum(policy_losses) / len(policy_losses))
        self.logs["value_loss"].append(sum(value_losses) / len(value_losses))
        self.logs["entropy"].append(sum(entropies) / len(entropies))
        self.logs["kl"].append(sum(kls) / len(kls))
        self.buffer.clear()

    def save(self, path):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "logs": self.logs,  # ← save all logged metrics
            "buffer_ptr": self.buffer.ptr
        }
        torch.save(checkpoint, path)
        print(f"[INFO] PPO agent saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore logs if available
        if "logs" in checkpoint:
            self.logs = checkpoint["logs"]
            print(f"[INFO] Training logs loaded).")
        else:
            print("[INFO] No training logs found in checkpoint.")

        # Restore buffer pointer (optional)
        if "buffer_ptr" in checkpoint:
            self.buffer.ptr = checkpoint["buffer_ptr"]

        print(f"[INFO] PPO agent loaded from {path}")