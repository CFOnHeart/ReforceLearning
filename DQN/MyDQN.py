# coding=utf-8

import sys
import os
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import Callback
import matplotlib.pyplot as plt

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class DQNAgent:
    def __init__(self, state_size, action_size,
                 render=False, load_model=False,
                 gamma=0.99, learning_rate=0.001,
                 epsilon=1.0, epsilon_decay=0.999,
                 epsilon_min=0.01, batch_size=64,
                 train_start=100, memory_size=2000,
                ):
        # env 的状态空间的设置
        self.state_size = state_size
        self.action_size = action_size

        # render表示是否打开gym下的动画展示,打开的话运行速度会大幅减慢
        self.render = render
        # load_model=True表示从文件中加载model
        self.load_model = load_model

        # 接下来的都是DQN的超参
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay  # e-贪心的e值随着步骤不断减小的比例
        self.epsilon_min = epsilon_min  # e-贪心的e值减小到一个阈值不再减小

        self.train_start = train_start
        self.batch_size = batch_size

        # 记忆数据存储模块
        self.memory = deque(maxlen=memory_size)

        # 初始化模型
        self.model = self.build_model()

        # 记录损失值
        self.history = LossHistory()
        self.losses_list = []

    def build_model(self, units=128):
        model = Sequential()
        model.add(Dense(units, input_dim=self.state_size,
                        activation='sigmoid', kernel_initializer='he_uniform'))
        model.add(Dense(units, activation='sigmoid',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def add_memory(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        min_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = min_batch[i][0]
            action.append(min_batch[i][1])
            reward.append(min_batch[i][2])
            done.append(min_batch[i][3])
            update_target[i] = min_batch[i][4]

        target = self.model.predict(update_input, batch_size=batch_size)
        target_val = self.model.predict(update_target, batch_size=batch_size)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * np.amax(target_val[i])

        self.model.fit(update_input, target, batch_size=batch_size, epochs=1, verbose=0, callbacks=[self.history])
        self.losses_list.append(self.history.losses[0])


def draw_score_plot(scores, filename='graph.png'):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('mean score')
    ax1.plot(range(len(scores)), scores, color='blue')
    plt.savefig(filename)


def draw_plot(scores, losses, filename='graph.png'):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('mean score')
    ax1.plot(range(len(scores)), scores, color='blue')


    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('mean loss-reward')
    ax2.plot(range(len(losses)), losses, color='blue')
    plt.savefig(filename)



