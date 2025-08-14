import copy
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

# 경험 재생 버퍼
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done

# Q 네트워크
class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

# DQN 에이전트
class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 3  # 0: 왼쪽, 1: 정지, 2: 오른쪽

        # Epsilon-greedy 설정
        self.epsilon_start = 0.2
        self.epsilon_end = 0.05
        self.epsilon_decay = 1
        self.epsilon = self.epsilon_start

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

# 학습 설정
episodes = 100
sync_interval = 20
env = gym.make('MountainCar-v0', render_mode='rgb_array')
agent = DQNAgent()
reward_history = []
success_count = 0

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, _, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        # 탄력 기반 보상: 위치 + 속도 + 도착 보상
        position, velocity = next_state
        reward = (position + 0.5) * 2.0 + abs(velocity) * 100.0
        if terminated:
            reward += 200

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if terminated:
        success_count += 1

    if episode % sync_interval == 0:
        agent.sync_qnet()

    agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
    reward_history.append(total_reward)

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

print(f"목표 도달 횟수: {success_count} / {episodes}")

# 리워드 그래프
plt.plot(range(episodes), reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN with Velocity-Based Reward (MountainCar)")
plt.show()

# 테스트 (탐욕 정책)
env2 = gym.make('MountainCar-v0', render_mode='human')
agent.epsilon = 0
state = env2.reset()[0]
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, terminated, truncated, info = env2.step(action)
    done = terminated | truncated
    state = next_state
    total_reward += reward
    env2.render()
print('Total Reward:', total_reward)
