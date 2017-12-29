# coding=utf-8

from MyQLearning import *
import gym
import math
import numpy as np
from numpy import mean, ptp, var, std
import matplotlib.pyplot as plt

# init env variables
state_size = [4, 4, 4, 4, 10, 10]
observation_high = [1.01, 1.01, 1.01, 1.01, 12.6, 28.3]
observation_low = [-1.01, -1.01, -1.01, -1.01, -12.6, -28.3]

# init QLearning parameters
n_states = 1
for val in state_size:
    n_states *= val
actions = [0, 1, 2]
epsilon = 0.9
alpha = 0.3
gamma = 0.95

# experiment requirments
max_episodes = 100
max_iter = 2000

model_name = './model/acrobot_model'
env_name = "Acrobot-v1"

# plot 
train_graph_name = './graph/acrobot_v1_train.png'
test_graph_name = './graph/acrobot_v1_test.png'

def run(graph_name, model_name, is_train, 
    max_episodes = 500, max_iter = 2000, encode_func=encode_value_uniform):
    # 初始化一个QLearning的结构
    if is_train == True:
        RL = QLearning(n_states, actions, epsilon=epsilon, alpha=alpha, gamma=gamma)
    else:
        RL = QLearning(n_states, actions, epsilon=1.0)

    RL.read_model(model_name)

    # 创建环境
    env = gym.make(env_name)
    env = env.unwrapped   # 不做这个会有很多限制,比如这里就无法访问这个env中的x_threshold属性(还不知道为什么)
    # env.metadata['video.frames_per_second'] = 1
    # 用一个变量记录每一轮的timesteps
    iters = []

    for i_episode in range(max_episodes):
        observation = env.reset()
        s = encode_state(observation, observation_high, observation_low, state_size)
        iter = 0
        flag = False
        while True:
            iter += 1
            # env.render()
            action = RL.choose_action(s)
            observation_, reward, done, info = env.step(action)

            if done:
                s_ = 'terminal'
            else:
                s_ = encode_state(observation_, observation_high, observation_low, state_size)

            # RL learn from this transition
            RL.learn(s, action, reward, s_)

            # swap  
            s = s_

            # break while loop when end of this episode
            if done or iter >= max_iter:
                iters.append(iter)
                print("episode:{}; iter: {}".format(i_episode, iter))
                break
        if flag == True:
            break

    iters_ar = np.array(iters)
    print("mean: {}; var: {}; std: {}".format(np.mean(iters_ar), np.var(iters_ar), np.std(iters_ar)))

    if is_train == True:
        RL.save_model(model_name)

    # draw plot 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("mean iter")
    ax.plot(range(len(iters)), iters)
    fig.savefig(graph_name)
    plt.show()


if __name__ == '__main__':
    # train and test func = uniform
    run('./graph/acrobot_v0_unifrom_train.png', './model/acrobot_v0_unifrom_model',
        True, max_episodes = 2000, max_iter=2000, encode_func=encode_value_uniform)
    run('./graph/acrobot_v0_unifrom_test.png', './model/acrobot_v0_unifrom_model',
        False, max_episodes = 500, max_iter=2000, encode_func=encode_value_uniform)

    # train and test func = root
    run('./graph/acrobot_v0_root_train.png', './model/acrobot_v0_root_model',
        True, max_episodes = 2000, max_iter=2000, encode_func=encode_value_root)
    run('./graph/acrobot_v0_root_test.png', './model/acrobot_v0_root_model',
        False, max_episodes = 500, max_iter=2000, encode_func=encode_value_root)

    # train and test func = square
    run('./graph/acrobot_v0_square_train.png', './model/acrobot_v0_square_model',
        True, max_episodes = 2000, max_iter=2000, encode_func=encode_value_square)
    run('./graph/acrobot_v0_square_test.png', './model/acrobot_v0_square_model',
        False, max_episodes = 500, max_iter=2000, encode_func=encode_value_square)
