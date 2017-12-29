# coding=utf-8

from MyQLearning import *
import gym
import math
import numpy as np
from numpy import mean, ptp, var, std
import matplotlib.pyplot as plt

# init env variables
state_size = [5, 10, 10, 10]
observation_high = [2.4, 5.0, 0.42, 5.0]
observation_low = [-2.4, -5.0, -0.42, -5.0]

# init QLearning parameters
n_states = 1
for val in state_size:
    n_states *= val
actions = [0, 1]
epsilon = 0.99
epsilon_decacy = 0.999
epsilon_min = 0.01
alpha = 0.3
gamma = 0.95

# experiment requirments
max_episodes = 500000
max_iter = 20000

model_name = './model/cartpole_model'
env_name = "CartPole-v0"

# plot 
graph_train_name = './graph/cartpole_v0_root_train.png'
graph_test_name = './graph/cartpole_v0_root_test.png'

def run(graph_name, is_train, max_episodes=50000, 
        max_iter=20000, encode_value_func=encode_value_uniform):  # is_train=False->test  is_train=True->train
    # 初始化一个QLearning的结构
    if is_train:
        RL = QLearning(n_states, actions, epsilon=epsilon, 
            epsilon_decacy=epsilon_decacy, epsilon_min=epsilon_min,
             alpha=alpha, gamma=gamma)
    else:
        RL = QLearning(n_states, actions, epsilon=0.0)
    RL.read_model(model_name)

    # 创建环境
    env = gym.make(env_name)
    env = env.unwrapped   # 不做这个会有很多限制,比如这里就无法访问这个env中的x_threshold属性(还不知道为什么)
    # env.metadata['video.frames_per_second'] = 1
    # 用一个变量记录每一轮的timesteps

    iters = []

    for i_episode in range(max_episodes):
        observation = env.reset()
        s = encode_state(observation, observation_high, observation_low, 
            state_size, encode_value_func=encode_value_func)
        iter = 0

        if is_train == True and len(iters) > 5:
            if np.mean(iters[-5:]) > 10000:
                break
        while True:
            iter += 1
            # env.render()
            action = RL.choose_action(s)
            observation_, reward, done, info = env.step(action)

            a, b, c, d = observation_
            r1 = (4.8 - abs(a)) / 4.8
            r2 = (5.0 - abs(b)) / 5.0
            r3 = (4.2 - abs(c)) / 4.2
            r4 = (5.0 - abs(d)) / 5.0
            reward = r1 + r2 + r3 + r4
            if done:
                s_ = 'terminal'
                reward = -100
            else:
                s_ = encode_state(observation_, observation_high, observation_low, 
                    state_size, encode_value_func=encode_value_func)

            # RL learn from this transition
            if is_train:
                RL.learn(s, action, reward, s_)

            # swap  
            s = s_

            # break while loop when end of this episode
            if done or iter >= max_iter:
                iters.append(iter)
                RL.alpha = (16.94 - math.pow(iter, 1.0/3.5)) / 16.94 * 0.20
                print("(episode:{}; iter: {}; epsilon: {}; alpha: {})".format(i_episode, iter, RL.epsilon, RL.alpha))
                break
        if i_episode % 5000 == 0:
            if is_train:
                RL.save_model(model_name)

    iters_ar = np.array(iters)
    print("mean: {}; var: {}; std: {}".format(np.mean(iters_ar), np.var(iters_ar), np.std(iters_ar)))

    if is_train:
        RL.save_model(model_name)

    # draw plot 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("mean iter")
    ax.plot(range(len(iters)), iters)
    fig.savefig(graph_name)

if __name__ == '__main__':
    # train 
    # run(graph_train_name, True)
    # test
    #run(graph_test_name, False, 
    #    max_episodes=1000, max_iter=20000, encode_value_func=encode_value_root)
    env = gym.make(env_name)
    print(env.observation_space.high)
    print(env.observation_space.low)
    print(env.action_space.n)
   