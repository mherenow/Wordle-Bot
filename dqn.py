import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class WordleState:
    def __init__(self, board_size = (6, 5)):
        self.board = np.zeros((board_size[0], board_size[1], 26))
        self.colors = np.zeros((board_size[0], board_size[1], 3))
        self.current_row = 0
        self.keyboard_state = np.ones(26)

    def encode_letters(self, letter):
        if letter == " ":
            return np.zeros(26)
        position = ord(letter.upper()) - ord('A')
        encoding = np.zeros(26)
        encoding[position] = 1
        return encoding
    
    def update(self, guess, feedback):
        for i, letter in enumerate(guess):
            self.board[self.current_row, i] = self.encode_letters(letter)

        for i, (status, _) in enumerate(feedback):
            if status == "correct":
                self.colors[self.current_row, i] = [1, 0, 0] #Green
            elif status == "present":
                self.colors[self.current_row, i] = [0, 1, 0] #Yellow
            else:
                self.colors[self.current_row, i] = [0, 0, 1] #Red

        for letter, (status, _) in zip(guess, feedback):
            idx = ord(letter.upper()) - ord('A')
            if status == 'absent':
                self.keyboard_state[idx] = 0

    def get_state_tensor(self):
        board_flat = self.board.flatten()
        colors_flat = self.colors.flatten()
        return np.concatenate([board_flat, colors_flat, self.keyboard_state])
    
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class WorldeDQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size

        #DQN networks
        self.policy_net = DQN(state_size, hidden_size, action_size)
        self.target_net = DQN(state_size, hidden_size, action_size) 
        self.target_net.load_state_dict(self.policy_net.state_dict())

        #Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10

        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)

    def select_action(self, state, valid_words):
        if random.random() < self.epsilon:
            return random.choice(valid_words)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)

            valid_indices = [i for i, word in enumerate(valid_words)]
            valid_q_values = q_values[0][valid_indices]

            max_idx = torch.argmax(valid_q_values).item()
            return valid_words[max_idx]
        
    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.FloatTensor(batch[0]).to(device)
        action_batch = torch.LongTensor(batch[1]).to(device)
        reward_batch = torch.FloatTensor(batch[2]).to(device)
        next_state_batch = torch.FloatTensor(batch[3]).to(device)
        done_batch = torch.FloatTensor(batch[4]).to(device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_agent(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, env.valid_words)
            next_state, reward, done = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.train(agent.batch_size)

        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        print(f"Episode: {episode + 1}, Reward: {total_reward}, epsilon: {agent.epsilon:.2f}")