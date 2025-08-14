import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from gym.wrappers import AtariPreprocessing, FrameStack, RecordVideo
import matplotlib.pyplot as plt
import os
np.bool8 = bool

# --- 환경 구성 ---
env = gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0, render_mode="rgb_array")
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
env = FrameStack(env, 4)

# 비디오 저장 경로
video_dir = "./pong_videos"
os.makedirs(video_dir, exist_ok=True)
env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: x == 999)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DQN 네트워크 정의 ---
class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device)
        )

# --- 하이퍼파라미터 ---
gamma = 0.99
lr = 1e-4
epsilon_start = 1.0
epsilon_final = 0.05
epsilon_decay = 500000
buffer_size = 100000
batch_size = 32
sync_target_steps = 1000
num_episodes = 1000

# --- 모델 및 기타 준비 ---
action_dim = env.action_space.n
policy_net = DQN(action_dim).to(device)
target_net = DQN(action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer(buffer_size)

steps_done = 0
episode_rewards = []

# --- 학습 루프 ---
for episode in range(num_episodes):
    obs = env.reset()[0]
    state = torch.tensor(obs, dtype=torch.float32).to(device)
    total_reward = 0
    done = False

    while not done:
        steps_done += 1
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * \
                  np.exp(-1. * steps_done / epsilon_decay)

        # 행동 선택
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(state.unsqueeze(0))
                action = q_values.argmax().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.tensor(next_obs, dtype=torch.float32).to(device)

        replay_buffer.add(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
        state = next_state
        total_reward += reward

        # 학습
        if len(replay_buffer.buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            q_values = policy_net(states)
            q_val = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                target_q = target_net(next_states).max(1)[0]
                target = rewards + (1 - dones) * gamma * target_q

            loss = nn.MSELoss()(q_val, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 타겟 네트워크 동기화
        if steps_done % sync_target_steps == 0:
            target_net.load_state_dict(policy_net.state_dict())

    episode_rewards.append(total_reward)
    if episode % 10 == 0:
        print(f"[Episode {episode}] Total Reward: {total_reward:.1f}")

# --- 보상 그래프 ---
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Pong Reward")
plt.show()

# --- 마지막 영상 경로 출력 ---
print(f"📹 마지막 영상은 {video_dir} 폴더에 저장되었습니다.")
