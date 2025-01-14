import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)

# NN for policy and value function
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        logits = self.fc(state)  # Output logits for categorical distribution
        return logits
    
class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.fc(state)


class ProjectAgent:
    def __init__(self, state_dim=6, action_dim=4, lr=1e-3, gamma=0.99, clip_eps=0.2, entropy_coeff=0.0):
        self.action_dim = action_dim
        
        self.mean_state = np.array([8.71713173e+05, 4.44882828e+02, 7.46098854e+02, 2.44172803e+01, 2.17874874e+03, 3.22276867e+04])
        self.std_state = np.array([2.04371166e+05, 3.55292671e+03, 3.17949758e+02, 1.12591133e+01, 1.77395840e+04, 1.77111172e+04])

        self.policy = Policy(state_dim, action_dim)
        self.value = Value(state_dim)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff


        self.mean_rwd = 15.73507607215155
        self.std_rwd = 2.45736898364524

    def scale(self, x):
        return np.where(x>1, np.log(x), x)

    def compute_returns(self, rewards, dones):
        returns = []
        g = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                g = 0
            rwd = self.scale(r) 

            g = (rwd-self.mean_rwd)/self.std_rwd + self.gamma * g
            returns.insert(0, g)
        return returns


    def update(self, states, actions, log_probs_old, returns):
        states = torch.tensor(((states-self.mean_state)/self.std_state), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)  # Discrete actions
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Policy update
        for _ in range(10):  # Number of policy optimization steps
            logits = self.policy(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratios = torch.exp(log_probs - log_probs_old)
            advantages = returns - self.value(states).detach().squeeze()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        # Value function update
        for _ in range(10):  # Number of value function optimization steps
            value_loss = nn.MSELoss()(self.value(states).squeeze(), returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

    def get_action(self, state, use_random=False, use_best = False):
        state = torch.tensor(((state-self.mean_state)/self.std_state), dtype=torch.float32).unsqueeze(0)
        logits = self.policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        if use_random==False:
            if use_best:
                action = torch.argmax(logits)
            else:
                action = dist.sample()
        else:
            action = torch.randint(0, self.action_dim, (1,))
        log_prob = dist.log_prob(action).item()
        return action.item(), log_prob

    def act(self, observation, use_random=False):
        state = torch.tensor(((observation-self.mean_state)/self.std_state), dtype=torch.float32)
        logits = self.policy(state)
        action = torch.argmax(logits).item()
        return action
        
    def save(self, path):
        torch.save(self.policy.state_dict(), "policy.pth")
        torch.save(self.value.state_dict(), "value.pth")

    def load(self):
        self.policy.load_state_dict(torch.load("policy.pth", weights_only=True, map_location=torch.device('cpu')))
        self.value.load_state_dict(torch.load("value.pth",  weights_only=True, map_location=torch.device('cpu')))
          
def plot_rewards(train_rewards, val_rewards_fixed, val_rewards_rand):
    """
    Plot training rewards.
    """
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(10, 6))
    plt.plot(train_rewards, label="Training Reward")
    plt.plot(val_rewards_fixed, label="Validation Reward Fixed")
    plt.plot(val_rewards_rand, label="Validation Reward Random")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.yscale("log")
    plt.title("Training and Validation Rewards")
    plt.legend()
    plt.grid()

    # Create logs directory if it does not exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
    plt.savefig("./logs/rewards_plot.png")
    plt.show()          

#class DQN(nn.Module):
#    def __init__(self, input_size=6, output_size=4): 
#        super(DQN, self).__init__()
#        self.fc1 = nn.Linear(input_size, 128)
#        self.fc2 = nn.Linear(128, 128)
#        self.fc3 = nn.Linear(128, output_size)

#    def forward(self, x):
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x
"""
    def train(self, env, max_episode):
        device = self.device
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state, device)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            if done:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return




    


""" 
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--train", type = bool, default = False)
#    args = parser.parse_args()
    
#    agent = ProjectAgent()
#    if args.train:
#        print("Training the agent")
#        agent.train(env, max_episode=300)
#        agent.save(agent.save_path)
#        print("Agent trained and saved")
         
