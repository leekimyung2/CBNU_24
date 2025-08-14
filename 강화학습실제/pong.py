import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import torchvision.transforms as T
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
import os

device = torch.device("cpu")

# 환경 설정
save_dir = "./videos"
os.makedirs(save_dir, exist_ok=True)
env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=4)
env = RecordVideo(env, video_folder=save_dir, episode_trigger=lambda e: True)

obs_shape = (84, 84)
n_actions = env.action_space.n

# 이미지 전처리
transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize(obs_shape),
    T.ToTensor()
])
def preprocess(obs):
    return transform(obs).squeeze(0)

# 개선된 VGG-style CNN 기반 Actor-Critic 모델
class ActorCritic(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Flatten()
        self.shared = nn.Linear(64 * 21 * 21, 512)
        self.actor = nn.Linear(512, action_size)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = self.fc(x)
        x = torch.relu(self.shared(x))
        return self.actor(x), self.critic(x)

model = ActorCritic(n_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)  # 학습률 감소

# 학습 기록
reward_list = []
epsilon_list = []
epsilon = 0.1  # 고정된 탐험 비율

def train(num_episodes=1000, gamma=0.99):
    global epsilon

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        state_queue = deque([preprocess(state)] * 4, maxlen=4)
        episode_reward = 0
        advantages = []

        done = False
        while not done:
            stacked_state = torch.stack(list(state_queue), dim=0).unsqueeze(0).to(device)
            logits, value = model(stacked_state)
            probs = torch.softmax(logits, dim=1)

            # ε-greedy: 확률 기반 or 무작위
            if np.random.rand() < epsilon:
                action = torch.tensor([[env.action_space.sample()]])
                log_prob = torch.tensor(0.0)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            episode_reward += reward

            next_processed = preprocess(next_state)
            state_queue.append(next_processed)

            with torch.no_grad():
                next_stacked = torch.stack(list(state_queue), dim=0).unsqueeze(0).to(device)
                _, next_value = model(next_stacked)
                target = reward + (0 if done else gamma * next_value.item())

            advantage = target - value.item()
            advantages.append(advantage)

            actor_loss = -log_prob * advantage
            critic_loss = (target - value) ** 2
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()

            # ✅ Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        # ✅ Advantage 정규화 (에피소드 단위로)
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        reward_list.append(episode_reward)
        epsilon_list.append(epsilon)

        print(f"[Ep {episode}] Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

    env.close()

    # 그래프 출력
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color='tab:blue')
    ax1.plot(reward_list, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Epsilon", color='tab:red')
    ax2.plot(epsilon_list, color='tab:red', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title("Reward & Epsilon")
    plt.tight_layout()
    plt.savefig("reward_epsilon_plot.png")
    plt.show()

train()
