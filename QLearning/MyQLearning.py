# coding=utf-8

import numpy as np
import pandas as pd
import time
import os
import math

def encode_value_uniform(value, high, low, state):
    return int((value - low) / (high - low) * state)

def encode_value_root(value, high, low, state):
    more = 0
    mid = (high+low)/2
    l = (high-low)/2
    value -= mid
    if value < 0:
        more = state // 2
        value = -value
    return int(more + math.sqrt(value) * state / 2 / math.sqrt(l))

def encode_value_square(value, high, low, state):
    more = 0
    mid = (high+low)/2
    l = (high-low)/2
    value -= mid
    if value < 0:
        more = state // 2
        value = -value
    # if int(more + value * value * state / 2 / (l * l)) >= state:
    #     print tmp, ' ', mid, ' ', l, ' ', int(more + value * value * state / 2 / (l * l)) 
    return int(more + value * value * state / 2 / (l * l))

def encode_state(observation, observation_high, observation_low, state_size, encode_value_func=encode_value_uniform):
    n = len(observation)
    state = 0
    for i in range(n):
        s = encode_value_func(observation[i], observation_high[i], observation_low[i], state_size[i])
        for j in range(i+1, n, 1):
            s *= state_size[j]
        state += s
    return state


class QLearning:
    def __init__(self, n_states, actions, epsilon=0.99, epsilon_decacy=0.999, epsilon_min=0.1, alpha=0.1, gamma=0.9):
        self.Q = pd.DataFrame(
            np.zeros((n_states, len(actions))),  # 将初始的Q值都设置为0
            columns=actions,  # Q表的列名
        )
        self.actions = actions
        self.n_actions = len(actions)
        self.n_states = n_states
        self.epsilon = epsilon  # epsilon-贪心算法,QLearning采样选择下一步的过程中用到
        self.epsilon_decacy = epsilon_decacy
        self.epsilon_min = epsilon_min
        self.alpha = alpha  # 更新步长(学习率)
        self.gamma = gamma  # 奖赏折扣

    # 根据当前状态按照, 传入的state表示行的标号,是个整数
    def choose_action(self, state):  
        # print('state: ', state)
        state_actions = self.Q.iloc[state, :]
        if (np.random.uniform() < self.epsilon) or (state_actions.all() == 0):  # 根据epsilon贪心算法,随机选择
            action_index = np.random.random_integers(self.n_actions)
            action_name = self.actions[action_index-1]
        else:
            action_name = state_actions.idxmax()

        return action_name

    def learn(self, s, action, reward, s_):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decacy
        q_predict = self.Q.ix[s, action]
        if s_ != 'terminal':
            if(s_ >= self.n_states or s_ < 0):
                print("error state : ", s_, self.n_states)
            q_target = reward + self.gamma * self.Q.iloc[s_, :].max()
        else:  # 游戏结束,不会有后续的步骤增加reward了
            q_target = reward

        self.Q.ix[s, action] += self.alpha * (q_target - q_predict)  # update Q table


    def save_model(self, filename='model'):
        fw = open(filename, mode='w')
        (n, m) = self.Q.shape
        print('start to save model to file')
        for i in range(n):
            line = ''
            for j in range(m):
                line = line + str(self.Q.iat[i, j]) + ' '
            line += '\n'
            fw.write(line)
        fw.close()
        print('model save to file successfully')


    def read_model(self, filename='model'):
        if os.path.exists(filename) == True:
            print('start to read model')
            fr = open(filename, mode='r')
            (n, m) = self.Q.shape
            i = 0 
            for line in fr.readlines():
                nums = line.split(' ')
                for j in range(len(nums)-1):
                    self.Q.ix[i, j] = float(nums[j])
                i += 1
            fr.close()
            print('model read successfully')
        else:
            print(filename + "doesn't exist")


