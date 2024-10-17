from . import NeuralNetwork as dn
import numpy as np
import random
import torch
from collections import deque
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from . import Snake as g
class Agent:
    def __init__(self, learning_rate, epsilon, epsilon_decay, epsilon_min, gamma, batch_size, replay_size):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = deque(maxlen=replay_size)
        
        self.main_NN = dn.NN(13, 4).to(self.device)
        self.target_NN = dn.NN(13, 4).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer_main = optim.Adam(self.main_NN.parameters(), lr=self.learning_rate)
        self.update_target_NN()
    def update_target_NN(self):
        self.target_NN.load_state_dict(self.main_NN.state_dict())
    def save_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    def get_batch_from_buffer(self):
        return random.sample(self.replay_buffer, self.batch_size)
    def save_model(self, path):
        torch.save(self.main_NN.state_dict(), path)
    def load_model(self, path):
        self.main_NN.load_state_dict(torch.load(path)) 
        self.update_target_NN()
    def choose_action(self, state):
        if np.random.uniform(0, 1) <= self.epsilon:
            return random.randint(0, 3)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # 1 x imput_size
        with torch.no_grad():
            q_values = self.main_NN(state)
        return np.argmax(q_values.cpu().numpy())
    def train_one_bacth(self):
        if len(self.replay_buffer) < self.batch_size:
            return 
        mini_batch = self.get_batch_from_buffer()
        state, action, reward, next_state, done = zip(*mini_batch) # map value from mini_batch
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.LongTensor(done).to(self.device)

        q_values = self.main_NN(state)
        next_q_values = self.target_NN(next_state)

        q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1) # map q_values with action
        next_q_values = next_q_values.max(1)[0]
        target_q_values = reward + (self.gamma * next_q_values * (1 - done))

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer_main.zero_grad()
        loss.backward()
        self.optimizer_main.step()
